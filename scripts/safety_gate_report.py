"""Count what the SafetyGate decided, separately from what DIET decided.

`rasa test nlu` scores the whole pipeline, so a probe-set result for
`ask_regulated_advice` or `insult` mixes two components: DIET (statistical)
and SafetyGate (deterministic regex, components/safety_gate.py). The lecturer
lists "separating safety evaluation from routing evaluation" as an
improvement, and notes that "unsafe requests should be evaluated through the
Safety Gate rather than the Intent Router" (LLM components notebook, §22).

For every dev-set message this script records
  - DIET's decision: the model with the gate switched off, so DIET +
    FallbackClassifier exactly as configured;
  - the gate's decision: components.safety_gate.classify() on the same text,
    with the patterns file and max_chars from config.yml;
  - the final decision: the gate's intent if it fired, else DIET's. The gate
    runs last in the pipeline and overrides, so this is what the bot does.
The model is then loaded a second time with the gate switched on, and the
final decisions are compared with the real pipeline as a check.

Dev sets only (tests/nlu_regression.yml, tests/offtopic_probe.yml), never the
final test set. Empty and over-long input (invalid_input) does not occur in
these sets; it is covered by tests/test_actions.py and the smoke test.

Usage:
    python scripts/safety_gate_report.py models/<model>.tar.gz
"""
from __future__ import annotations

import asyncio
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

REPO_ROOT = Path(__file__).resolve().parent.parent
# The `rasa` CLI puts the project root on sys.path, a plain script does not.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dialogue_smoke import quiet_rasa_logs  # noqa: E402
from threshold_sweep import PROBE_FILE, REGRESSION_FILE, load_cases  # noqa: E402

import yaml  # noqa: E402

from components import safety_gate  # noqa: E402

GateDecision = Optional[Tuple[str, str]]   # (category, intent) or None


def gate_settings() -> Tuple[Path, int]:
    """patterns_file and max_chars of the SafetyGate entry in config.yml."""
    config = yaml.safe_load((REPO_ROOT / "config.yml").read_text())
    for component in config.get("pipeline", []):
        if component.get("name", "").endswith("SafetyGate"):
            defaults = safety_gate.SafetyGate.get_default_config()
            path = Path(component.get("patterns_file", defaults["patterns_file"]))
            if not path.is_absolute():
                path = REPO_ROOT / path
            return path, int(component.get("max_chars", defaults["max_chars"]))
    raise SystemExit("SafetyGate is not in the config.yml pipeline.")


async def parse_all(model: str, texts: List[str], gate_on: bool) -> List[str]:
    """Final intent per text, with the gate switched on or off."""
    from rasa.core.agent import Agent

    original = safety_gate.SafetyGate.process
    if not gate_on:
        safety_gate.SafetyGate.process = lambda self, messages: messages
    try:
        quiet_rasa_logs()
        agent = Agent.load(model)
        quiet_rasa_logs()  # loading a model can reset the configuration
        intents = []
        for text in texts:
            result = await agent.parse_message(text)
            intents.append((result.get("intent") or {}).get("name"))
        return intents
    finally:
        safety_gate.SafetyGate.process = original


def report(title: str, rows: List[Dict]) -> None:
    n = len(rows)
    fired = [r for r in rows if r["gate"]]
    print(f"\n=== {title}  (n = {n})")
    print(f"Gate fired on {len(fired)} of {n} messages.")

    by_category = Counter(r["gate"][0] for r in fired)
    for category, count in sorted(by_category.items()):
        hits = [r for r in fired if r["gate"][0] == category]
        intent = hits[0]["gate"][1]
        correct = [r for r in hits if r["expected"] == intent]
        rescued = [r for r in correct if r["diet"] != intent]
        print(f"  {category} -> {intent}: {count} fired | correct {len(correct)} "
              f"(DIET alone already right {len(correct) - len(rescued)}, "
              f"rescued by the gate {len(rescued)}) | "
              f"false positives {count - len(correct)}")
        for r in rescued:
            print(f"      rescued: {r['text']!r} (DIET said {r['diet']})")
        for r in hits:
            if r["expected"] != intent:
                print(f"      FALSE POSITIVE: {r['text']!r} (expected {r['expected']})")

    gated_intents = sorted({r["gate"][1] for r in fired} | GATED_INTENTS)
    for intent in gated_intents:
        expected = [r for r in rows if r["expected"] == intent]
        if not expected:
            continue
        silent = [r for r in expected if not r["gate"]]
        silent_right = sum(1 for r in silent if r["diet"] == intent)
        final_right = sum(1 for r in expected if r["final"] == intent)
        print(f"  expected {intent} (n = {len(expected)}): gate silent on "
              f"{len(silent)}, DIET right on {silent_right} of those | "
              f"recall DIET alone {sum(1 for r in expected if r['diet'] == intent)}"
              f"/{len(expected)}, DIET + gate {final_right}/{len(expected)}")
        for r in silent:
            if r["diet"] != intent:
                print(f"      missed by both: {r['text']!r} (DIET said {r['diet']})")

    diet_acc = sum(1 for r in rows if r["diet"] == r["expected"])
    final_acc = sum(1 for r in rows if r["final"] == r["expected"])
    # Strict: the exact expected intent. An off_topic message sent to the
    # clarification flow (nlu_fallback) counts as wrong here, although the
    # threshold sweep treats both as "declined".
    print(f"  Exact-intent accuracy on this set: DIET alone {diet_acc}/{n} "
          f"({diet_acc / n:.3f}), DIET + gate {final_acc}/{n} ({final_acc / n:.3f})")


GATED_INTENTS: set = set()


async def main(model: str) -> int:
    patterns_path, max_chars = gate_settings()
    categories = safety_gate.load_categories(patterns_path)
    GATED_INTENTS.update(intent for _, intent, _ in categories)

    sets = {
        "tests/nlu_regression.yml": load_cases(REGRESSION_FILE),
        "tests/offtopic_probe.yml": load_cases(PROBE_FILE),
    }
    texts = [t for cases in sets.values() for t, _ in cases]

    diet = await parse_all(model, texts, gate_on=False)
    pipeline = await parse_all(model, texts, gate_on=True)

    print(f"model: {model}")
    print(f"patterns: {patterns_path.relative_to(REPO_ROOT)} "
          f"({', '.join(f'{name} -> {intent}' for name, intent, _ in categories)}); "
          f"max_chars {max_chars}")

    rows_all: List[Dict] = []
    i = 0
    for name, cases in sets.items():
        rows = []
        for text, expected in cases:
            gate: GateDecision = safety_gate.classify(text, categories, max_chars)
            rows.append({
                "text": text,
                "expected": expected,
                "diet": diet[i],
                "gate": gate,
                "final": gate[1] if gate else diet[i],
                "pipeline": pipeline[i],
            })
            i += 1
        report(name, rows)
        rows_all.extend(rows)
    report("both dev sets", rows_all)

    mismatched = [r for r in rows_all if r["final"] != r["pipeline"]]
    print(f"\nCheck against the real pipeline (gate on): "
          f"{len(rows_all) - len(mismatched)}/{len(rows_all)} identical.")
    for r in mismatched:
        print(f"  MISMATCH: {r['text']!r} computed {r['final']}, pipeline {r['pipeline']}")
    return 1 if mismatched else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/safety_gate_report.py models/<model>.tar.gz")
        sys.exit(2)
    sys.exit(asyncio.run(main(sys.argv[1])))
