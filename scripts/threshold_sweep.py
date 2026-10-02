"""Measure what FallbackClassifier's threshold would do, without retraining.

`FallbackClassifier` does not change how DIET classifies anything — it reads
DIET's ranking afterwards and replaces the prediction with `nlu_fallback` when
the top intent is below `threshold`, or when the top two are within
`ambiguity_threshold` of each other. So the effect of a different threshold can
be measured by parsing once and re-applying the rule at several cut-offs.

Two things are in tension:
  - Raise it, and confidently-wrong answers to out-of-scope messages turn into
    the clarification flow, which is the better answer.
  - Raise it too far, and correct answers to real requests get thrown away and
    the user is asked "did you mean…?" about something they said perfectly.

Usage:
    python scripts/threshold_sweep.py [model.tar.gz]

Reads the dev sets tests/nlu_regression.yml and tests/offtopic_probe.yml
(never the final test set) plus the out-of-scope list below, then prints a
table per threshold. Columns used for the threshold decision (D58):
  - request_human_advisor recall: a missed handover request is the costly
    error (NLP-W345 ticket-routing task; DESIGN_RATIONALE D53);
  - fallback rate: share of real requests sent to clarification;
  - macro F1 over all dev labels, where a fallback counts as "not answered
    as a task" (label off_topic).
Limits for the first two are fixed BEFORE the sweep is read; among the
thresholds that meet them, the best macro F1 is chosen (AI-MALL cell 56:
filter by explicit constraints first, then rank).
"""
from __future__ import annotations

import asyncio
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

REPO_ROOT = Path(__file__).resolve().parent.parent
REGRESSION_FILE = REPO_ROOT / "tests" / "nlu_regression.yml"
PROBE_FILE = REPO_ROOT / "tests" / "offtopic_probe.yml"
COSTLY_INTENT = "request_human_advisor"

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
AMBIGUITY_THRESHOLD = 0.1

# Messages this bot should not answer. These are deliberately NOT the phrases
# used to train the `off_topic` intent — a sweep run against its own training
# examples would only prove the model can memorise. "When did the Berlin wall
# fall" is in the list on purpose: it carries a city name, which is the kind of
# thing that tempts the model into an eco-travel intent.
OUT_OF_SCOPE = [
    "when did the Berlin wall fall",
    "how do I fix a flat tyre",
    "who wrote Hamlet",
    "recommend a good series to watch",
    "my laptop will not turn on",
    "translate good evening into Japanese",
    "set an alarm for seven in the morning",
    "what is the best pizza topping",
]


def load_cases(path: Path) -> List[Tuple[str, str]]:
    """Read (text, expected_intent) pairs from the regression YAML.

    Parsed by hand rather than with the YAML loader because the examples are a
    literal block where each line also carries entity annotations.
    """
    cases: List[Tuple[str, str]] = []
    intent = None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("- intent:"):
            intent = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("- ") and intent and line.startswith("    "):
            text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", stripped[2:])
            cases.append((text, intent))
    return cases


def diet_ranking(ranking: List[Dict]) -> List[Dict]:
    """DIET's own ranking, without what FallbackClassifier added to it.

    When the trained model's classifier fires, Rasa inserts `nlu_fallback`
    (confidence = the configured threshold) at the top of `intent_ranking`
    (rasa/nlu/classifiers/fallback_classifier.py). Reading that entry as if it
    were DIET's made every threshold below the model's own identical to it —
    the Phase 5 table's 0.3/0.4/0.5 rows were copies of the 0.6 row.
    """
    return [r for r in ranking if r["name"] != "nlu_fallback"]


def decide(ranking: List[Dict], threshold: float) -> str:
    """Apply FallbackClassifier's rule to one DIET ranking."""
    ranking = diet_ranking(ranking)
    if not ranking:
        return "nlu_fallback"
    top = ranking[0]
    if top["confidence"] < threshold:
        return "nlu_fallback"
    if len(ranking) > 1 and top["confidence"] - ranking[1]["confidence"] < AMBIGUITY_THRESHOLD:
        return "nlu_fallback"
    return top["name"]


async def main(model: str) -> int:
    from rasa.core.agent import Agent

    agent = Agent.load(model)
    # The regression file also holds `off_topic` examples. Those are messages
    # the bot should decline, so they belong with the out-of-scope cases —
    # counted as "real asks", every one handled correctly by the fallback
    # would be misreported as a lost request.
    # The probe set's `nlu_fallback` class (gibberish, other languages,
    # prompt injection) has no intent to be answered as, so it is
    # out-of-scope like `off_topic`.
    dev = load_cases(REGRESSION_FILE) + load_cases(PROBE_FILE)
    declined = ("off_topic", "nlu_fallback")
    in_scope = [(t, i) for t, i in dev if i not in declined]
    out_of_scope = list(dict.fromkeys(
        OUT_OF_SCOPE + [t for t, i in dev if i in declined]
    ))

    parsed_in: List[Tuple[str, str, List[Dict]]] = []
    for text, expected in in_scope:
        result = await agent.parse_message(text)
        parsed_in.append((text, expected, result.get("intent_ranking", [])))

    parsed_out: List[Tuple[str, List[Dict]]] = []
    for text in out_of_scope:
        result = await agent.parse_message(text)
        parsed_out.append((text, result.get("intent_ranking", [])))

    print(f"model: {model}")
    print(f"in-scope cases: {len(parsed_in)}   out-of-scope cases: {len(parsed_out)}")
    print(f"ambiguity_threshold held at {AMBIGUITY_THRESHOLD}\n")
    from sklearn.metrics import f1_score

    n_costly = sum(1 for _, e, _ in parsed_in if e == COSTLY_INTENT)
    print(f"{'threshold':>9} | {'right':>5} | {'wrong':>5} | {'to fallback':>11} | "
          f"{'fallback rate':>13} | {COSTLY_INTENT + ' recall':>28} | "
          f"{'off-topic handled':>17} | {'macro F1':>8}")
    print("-" * 120)

    for threshold in THRESHOLDS:
        right = wrong = lost = 0
        for _, expected, ranking in parsed_in:
            decision = decide(ranking, threshold)
            if decision == "nlu_fallback":
                lost += 1
            elif decision == expected:
                right += 1
            else:
                wrong += 1
        # An off-topic message is handled well either by being recognised as
        # `off_topic` (the bot explains its scope) or by falling through to
        # the clarification flow (which now also explains its scope). Being
        # answered as a real intent is the failure.
        caught = sum(
            1 for _, ranking in parsed_out
            if decide(ranking, threshold) in ("nlu_fallback", "off_topic")
        )
        costly_hit = sum(1 for _, e, r in parsed_in
                         if e == COSTLY_INTENT and decide(r, threshold) == e)
        # Macro F1: a fallback or off_topic answer means "declined".
        def label(decision: str) -> str:
            return "off_topic" if decision in declined else decision
        y_true = [e for _, e, _ in parsed_in] + ["off_topic"] * len(parsed_out)
        y_pred = ([label(decide(r, threshold)) for _, _, r in parsed_in]
                  + [label(decide(r, threshold)) for _, r in parsed_out])
        macro = f1_score(y_true, y_pred, labels=sorted(set(y_true)),
                         average="macro", zero_division=0)
        print(f"{threshold:>9.2f} | {right:>5} | {wrong:>5} | {lost:>11} | "
              f"{lost / len(parsed_in):>13.1%} | "
              f"{costly_hit:>21}/{n_costly:<6} | "
              f"{caught:>11}/{len(parsed_out):<5} | {macro:>8.3f}")

    print("\nOut-of-scope messages and what the model thinks they are:")
    for text, ranking in parsed_out:
        ranking = diet_ranking(ranking)
        if ranking:
            print(f"  {text:42s} -> {ranking[0]['name']:22s} {ranking[0]['confidence']:.2f}")

    print("\nIn-scope cases with the lowest confidence (these are what a higher "
          "threshold would cost you):")
    lowest = sorted(parsed_in, key=lambda c: diet_ranking(c[2])[0]["confidence"]
                    if diet_ranking(c[2]) else 0)[:6]
    for text, expected, ranking in lowest:
        ranking = diet_ranking(ranking)
        if ranking:
            print(f"  {text:42s} -> {ranking[0]['name']:22s} {ranking[0]['confidence']:.2f}"
                  f"   (expected {expected})")

    # The single number quoted in the docs: any threshold at or below it
    # loses none of the in-scope requests DIET gets right.
    correct = [(t, diet_ranking(r)[0]["confidence"]) for t, e, r in parsed_in
               if diet_ranking(r) and diet_ranking(r)[0]["name"] == e]
    if correct:
        text, conf = min(correct, key=lambda c: c[1])
        print(f"\nLowest-confidence CORRECT in-scope prediction: {conf:.2f} ({text!r}); "
              f"DIET correct on {len(correct)}/{len(parsed_in)} before any threshold")
    return 0


if __name__ == "__main__":
    model_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if not model_arg:
        models = sorted((REPO_ROOT / "models").glob("*.tar.gz"))
        if not models:
            print("No model found. Train one first with `rasa train`.")
            sys.exit(1)
        model_arg = str(models[-1])
    sys.exit(asyncio.run(main(model_arg)))
