"""Score what the bot actually decides on a held-out NLU file, at the configured threshold.

`rasa test nlu` reverts FallbackClassifier (rasa/nlu/test.py:1304-1308), so it scores
the classifier. This script asks the trained model itself (Agent.parse_message,
the code path the server uses), so FallbackClassifier at the configured
threshold and the SafetyGate are both applied, exactly as in a conversation.
No threshold is swept here; the threshold is whatever the model was trained with.

Columns follow the lecturer's ticket-triage table (NLP weeks 3-5 notebook, Advanced Task) and
scripts/threshold_sweep.py: routed, sent to clarification, accuracy of routed,
request_human_advisor recall (a clarification counts as a miss), fallback rate;
off_topic items count as handled when the bot declines them (off_topic or
clarification). Items whose nearest training sentence (TF-IDF char_wb 3-5,
cosine >= 0.80, same intent) is a near-copy are flagged, and every figure is
also given without them.

Only counts go to stdout. Per-item rows (text, expected, decision, top-2
confidences) are written to <out>/decisions.csv for the error table.

Usage:
    python scripts/final_test_decisions.py <model.tar.gz> <out folder> <nlu file> [<nlu file> ...]
"""
from __future__ import annotations

import asyncio
import csv
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import yaml  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dialogue_smoke import quiet_rasa_logs  # noqa: E402

COSTLY_INTENT = "request_human_advisor"
DECLINED = ("off_topic", "nlu_fallback")
NEAR_COPY = 0.80


def load(path: Path) -> List[Dict]:
    """(text, intent) per example, entity markup removed."""
    rows = []
    for block in yaml.safe_load(path.read_text())["nlu"]:
        if "intent" not in block:
            continue
        for line in block["examples"].splitlines():
            line = line.strip()
            if line.startswith("- "):
                rows.append({"text": re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line[2:]),
                             "expected": block["intent"]})
    return rows


def normalise(text: str) -> str:
    """Same normalisation as tests/test_actions.py::_nlu_examples and the
    leakage audit: lowercase, punctuation removed, spaces collapsed."""
    text = re.sub(r"[^\w\s<>]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def near_copy_flags(rows: List[Dict]) -> None:
    train = []
    for path in sorted((REPO_ROOT / "data").glob("nlu*.yml")):
        train += load(path)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit([normalise(t["text"]) for t in train])
    sim = cosine_similarity(vec.transform([normalise(r["text"]) for r in rows]),
                            vec.transform([normalise(t["text"]) for t in train]))
    for r, row in enumerate(rows):
        j = sim[r].argmax()
        row["near_copy"] = bool(sim[r, j] >= NEAR_COPY and train[j]["expected"] == row["expected"])


def summarise(rows: List[Dict]) -> str:
    in_scope = [r for r in rows if r["expected"] not in DECLINED]
    out_scope = [r for r in rows if r["expected"] in DECLINED]
    lost = sum(r["decision"] == "nlu_fallback" for r in in_scope)
    right = sum(r["decision"] == r["expected"] for r in in_scope)
    routed = len(in_scope) - lost
    costly = [r for r in in_scope if r["expected"] == COSTLY_INTENT]
    hit = sum(r["decision"] == COSTLY_INTENT for r in costly)
    handled = sum(r["decision"] in DECLINED for r in out_scope)
    acc = f"{right / routed:.3f} ({right}/{routed})" if routed else "n/a"
    rha = f"{hit / len(costly):.3f} ({hit}/{len(costly)})" if costly else "n/a"
    rate = f"{lost / len(in_scope):.1%}" if in_scope else "n/a"
    return (f"n {len(rows)} | in-scope {len(in_scope)}: routed {routed}, to clarification {lost}, "
            f"accuracy of routed {acc}, fallback rate {rate}, RHA recall {rha} | "
            f"off-topic handled {handled}/{len(out_scope)} | "
            f"overall exact decision accuracy {sum(r['decision'] == r['expected'] for r in rows)}/{len(rows)}")


async def main(model: str, out_dir: Path, files: List[Path]) -> int:
    from rasa.core.agent import Agent

    quiet_rasa_logs()
    agent = Agent.load(model)
    quiet_rasa_logs()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    lines = [f"model: {model}"]
    for path in files:
        rows = load(path)
        near_copy_flags(rows)
        for row in rows:
            result = await agent.parse_message(row["text"])
            ranking = [r for r in result.get("intent_ranking", []) if r["name"] != "nlu_fallback"]
            row["decision"] = (result.get("intent") or {}).get("name")
            row["top1"] = f"{ranking[0]['name']} {ranking[0]['confidence']:.3f}" if ranking else ""
            row["top2"] = f"{ranking[1]['name']} {ranking[1]['confidence']:.3f}" if len(ranking) > 1 else ""
            row["gate"] = result.get("safety_gate")
            row["file"] = path.name
        all_rows += rows
        clean = [r for r in rows if not r["near_copy"]]
        lines.append(f"{path.name}")
        lines.append(f"  all items      : {summarise(rows)}")
        lines.append(f"  no near-copies : {summarise(clean)} (near-copies removed: {len(rows) - len(clean)})")
        lines.append(f"  SafetyGate fired on {sum(1 for r in rows if r['gate'])} items, "
                     f"correct {sum(1 for r in rows if r['gate'] and r['decision'] == r['expected'])}")
    with open(out_dir / "decisions.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "text", "expected", "decision", "top1",
                                                    "top2", "gate", "near_copy"])
        writer.writeheader()
        writer.writerows(all_rows)
    text = "\n".join(lines)
    (out_dir / "decisions_summary.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    sys.exit(asyncio.run(main(sys.argv[1], Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]])))
