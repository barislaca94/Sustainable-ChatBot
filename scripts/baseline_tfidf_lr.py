"""A classical baseline for the NLU model: TF-IDF + logistic regression.

Why it exists: "Build a simple baseline before using complex models."
(NLP week 1 slides, slide 54). The question "why DIET?" has to be answered with numbers
measured under the same protocol, so this script trains on exactly the
training part of a `rasa data split` and scores exactly its test part, the
same files DIET was evaluated on (NLP week 7 notebook, §16: "A single stratified split is
reused so every model is evaluated on exactly the same tickets.").

Two feature sets, everything else held fixed (transformer fine-tuning notebook, Challenge 5, "Change
only one variable"):
  - word:      the lecturer's TF-IDF baseline as written in the NLP week 7 notebook, code cell
               83: lowercase, ngram_range=(1, 2), sublinear_tf=True, then
               LogisticRegression(max_iter=2000, random_state=42). This is
               the primary baseline.
  - word+char: the same, plus a char_wb TF-IDF with ngram_range=(3, 5), the
               setting of the NLP weeks 3-5 notebook, Additional Task 4. Secondary: DIET also
               sees character n-grams (config.yml), so this variant compares
               the classifiers with closer feature information.

The vectorizers live inside a scikit-learn Pipeline, so they are fitted on
the training texts only (NLP week 7 notebook, §16: "Using a pipeline prevents data leakage
because the test texts do not influence the TF-IDF vocabulary").

Scores are classifier-only, like `rasa test nlu` (no fallback threshold;
rasa/nlu/test.py:1304-1308), so the DIET column read from that run's
intent_report.json is directly comparable. Entity markup is stripped by
Rasa's own loader; entities are not part of this comparison.

Usage:
    python scripts/baseline_tfidf_lr.py <split folder> [<out folder>]

The split folder must contain data/training_data.yml, data/test_data.yml and
results/intent_report.json (DIET on the same test set).
"""
from __future__ import annotations

import csv
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402

RANDOM_STATE = 42
COSTLY_INTENT = "request_human_advisor"


def load_examples(path: Path) -> Tuple[List[str], List[str]]:
    """(texts, intents) from a Rasa NLU file, entity markup removed."""
    from rasa.shared.nlu.training_data.loading import load_data

    data = load_data(str(path))
    examples = [m for m in data.intent_examples if m.get("intent")]
    return [m.get("text") for m in examples], [m.get("intent") for m in examples]


def word_tfidf() -> TfidfVectorizer:
    # NLP week 7 notebook, code cell 83.
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)


def build(features: str) -> Pipeline:
    if features == "word":
        vectorizer = word_tfidf()
    else:
        vectorizer = FeatureUnion([
            ("word", word_tfidf()),
            # NLP weeks 3-5 notebook, Additional Task 4.
            ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb",
                                     ngram_range=(3, 5), sublinear_tf=True)),
        ])
    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])


def per_intent(report: Dict) -> Dict[str, Dict]:
    return {k: v for k, v in report.items()
            if isinstance(v, dict) and k not in ("macro avg", "weighted avg", "micro avg")}


def main(split_dir: Path, out_dir: Path) -> int:
    x_train, y_train = load_examples(split_dir / "data" / "training_data.yml")
    x_test, y_test = load_examples(split_dir / "data" / "test_data.yml")
    diet = json.loads((split_dir / "results" / "intent_report.json").read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict] = {}
    errors: List[Dict] = []
    for features in ("word", "word+char"):
        model = build(features)
        model.fit(x_train, y_train)
        y_pred = list(model.predict(x_test))
        # Same label set as Rasa's report: every label seen in truth or prediction.
        labels = sorted(set(y_test) | set(y_pred))
        report = classification_report(y_test, y_pred, labels=labels,
                                       output_dict=True, zero_division=0)
        report["accuracy"] = accuracy_score(y_test, y_pred)
        results[features] = report
        errors += [{"features": features, "text": t, "true": y, "pred": p}
                   for t, y, p in zip(x_test, y_test, y_pred) if y != p]

    (out_dir / "baseline_report.json").write_text(json.dumps(results, indent=2))
    with open(out_dir / "baseline_errors.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["features", "text", "true", "pred"])
        writer.writeheader()
        writer.writerows(errors)

    lines = [
        f"split: {split_dir}",
        f"train n = {len(x_train)}, test n = {len(y_test)}, intents in test = {len(set(y_test))}",
        "classifier-only scores on the same test set (no fallback threshold)",
        "",
        f"{'':24s} {'DIET':>12s} {'TF-IDF word':>12s} {'word+char':>12s}",
    ]
    columns = [diet, results["word"], results["word+char"]]
    lines.append(f"{'accuracy':24s} " + " ".join(f"{c['accuracy']:>12.3f}" for c in columns))
    lines.append(f"{'macro F1':24s} " + " ".join(f"{c['macro avg']['f1-score']:>12.3f}" for c in columns))
    lines.append(f"{'weighted F1':24s} " + " ".join(f"{c['weighted avg']['f1-score']:>12.3f}" for c in columns))
    costly = [per_intent(c).get(COSTLY_INTENT, {}) for c in columns]
    lines.append(f"{'RHA recall':24s} " + " ".join(f"{c.get('recall', 0):>12.3f}" for c in costly))
    lines += ["", "per-intent F1 (n = test support):",
              f"{'intent':24s} {'n':>3s} {'DIET':>6s} {'word':>6s} {'w+c':>6s}"]
    for intent in sorted(set(y_test)):
        f1s = [per_intent(c).get(intent, {}).get("f1-score", 0.0) for c in columns]
        n = per_intent(diet).get(intent, {}).get("support", 0)
        lines.append(f"{intent:24s} {n:>3d} " + " ".join(f"{f:>6.2f}" for f in f1s))
    text = "\n".join(lines)
    (out_dir / "baseline_summary.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    split = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else split / "baseline_tfidf_lr"
    sys.exit(main(split, out))
