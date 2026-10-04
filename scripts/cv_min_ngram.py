"""Measure DIET with char_wb min_ngram 2 against min_ngram 1, on identical folds.

Protocol and decision rule written down before the run. Same folds, metrics and
near-copy definition as scripts/cv_epochs.py. The min_ngram 1 arm is read from
that run's 100-epoch rows (identical folds, config and seed), so only
min_ngram 2 is trained here, each time with a fresh Rasa cache.

Also prints the char_wb vocabulary size at min_ngram 1-4 (Worksheet 9, Part 7), using
scikit-learn's CountVectorizer with the settings Rasa's CountVectorsFeaturizer
passes to it (char_wb, lowercase, max_ngram 4) on all training examples.

Usage (about 15 minutes):
    python scripts/cv_min_ngram.py [out folder]
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import RepeatedStratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cv_compare import load_all, write_fold  # noqa: E402
from cv_epochs import COSTLY_INTENT, NEAR_COPY, RUN, SEED, macro_f1, predict  # noqa: E402

EVAL = REPO_ROOT / "evaluation" / "phase7_2026-10-03"
REFERENCE_ROWS = EVAL / "cv_epochs_isometric-rower" / "epochs_rows.json"


def write_config(path: Path, min_ngram: int) -> None:
    """config.yml without the SafetyGate, char_wb min_ngram set, epochs 100."""
    config = yaml.safe_load((REPO_ROOT / "config.yml").read_text())
    config["pipeline"] = [c for c in config["pipeline"]
                          if not c["name"].endswith("SafetyGate")]
    for component in config["pipeline"]:
        if component["name"] == "CountVectorsFeaturizer" and component.get("analyzer") == "char_wb":
            component["min_ngram"] = min_ngram
        if component["name"] == "DIETClassifier":
            assert component["epochs"] == 100, "the epochs experiment kept 100 epochs"
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def main(out_dir: Path) -> int:
    from rasa.shared.nlu.training_data.loading import load_data

    data = load_all()
    examples = [m for m in data.training_examples if m.get("intent")]
    texts = [m.get("text") for m in examples]
    intents = [m.get("intent") for m in examples]
    original = {m.get("text") for m in load_data(str(REPO_ROOT / "data" / "nlu.yml")).training_examples
                if m.get("intent")}
    folds = list(RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=SEED)
                 .split(texts, intents))

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k, (train_idx, test_idx) in enumerate(folds):
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit([texts[i] for i in train_idx])
            sim = cosine_similarity(vec.transform([texts[i] for i in test_idx]),
                                    vec.transform([texts[i] for i in train_idx]))
            near = {i for r, i in enumerate(test_idx)
                    if sim[r].max() >= NEAR_COPY and intents[train_idx[sim[r].argmax()]] == intents[i]}

            work = tmp / f"fold{k}"
            work.mkdir()
            write_fold(data, [examples[i] for i in train_idx], work / "train.yml", True)
            write_fold(data, [examples[i] for i in test_idx], work / "test.yml", False)
            write_fold(data, [examples[i] for i in train_idx], work / "train_eval.yml", False)
            write_config(work / "config.yml", 2)
            env = {**os.environ, "RASA_CACHE_DIRECTORY": str(work / "cache")}
            started = time.time()
            subprocess.run(["rasa", "train", "nlu", "-c", str(work / "config.yml"),
                            "--nlu", str(work / "train.yml"), "--out", str(work),
                            "--fixed-model-name", "model"], env=env, **RUN)
            seconds = time.time() - started
            test_pred = predict(work / "model.tar.gz", work / "test.yml", work / "test")
            train_pred = predict(work / "model.tar.gz", work / "train_eval.yml", work / "train")
            if any(texts[i] not in test_pred for i in test_idx) or \
                    any(texts[i] not in train_pred for i in train_idx):
                raise SystemExit(f"fold {k}: predictions missing")

            y_true = [intents[i] for i in test_idx]
            y_pred = [test_pred[texts[i]] for i in test_idx]
            orig = [j for j, i in enumerate(test_idx) if texts[i] in original]
            clean = [j for j, i in enumerate(test_idx) if i not in near]
            rows.append({
                "fold": k, "min_ngram": 2, "train_seconds": round(seconds, 1),
                "macro_f1": macro_f1(y_true, y_pred),
                "macro_f1_original": macro_f1([y_true[j] for j in orig], [y_pred[j] for j in orig]),
                "macro_f1_no_near_copy": macro_f1([y_true[j] for j in clean], [y_pred[j] for j in clean]),
                "test_accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true),
                "train_accuracy": sum(train_pred[texts[i]] == intents[i] for i in train_idx) / len(train_idx),
                "rha_hits": sum(1 for t, p in zip(y_true, y_pred) if t == COSTLY_INTENT and p == t),
                "rha_total": sum(1 for t in y_true if t == COSTLY_INTENT),
            })
            print(f"fold {k}: {seconds:.0f}s", flush=True)
            (out_dir / "min_ngram2_rows.json").write_text(json.dumps(rows, indent=2))

    reference = [dict(r, min_ngram=1) for r in json.loads(REFERENCE_ROWS.read_text())
                 if r["epochs"] == 100]
    arms = {1: reference, 2: rows}

    def col(arm, key):
        return [r[key] for r in sorted(arms[arm], key=lambda r: r["fold"])]

    lines = ["DIET only, 100 epochs, same 9 folds; min_ngram 1 arm from cv_epochs (100 epochs)",
             f"{'min_ngram':>9s} {'macro F1':>15s} {'F1 original':>12s} {'F1 no near-copy':>16s} "
             f"{'test acc':>9s} {'train acc':>9s} {'RHA recall':>14s} {'train s':>8s}"]
    summary = {}
    for arm in (1, 2):
        hits, total = sum(col(arm, "rha_hits")), sum(col(arm, "rha_total"))
        f1 = col(arm, "macro_f1")
        summary[arm] = (statistics.mean(f1), hits / total)
        lines.append(f"{arm:>9d} {statistics.mean(f1):.3f} ({statistics.stdev(f1):.3f}) "
                     f"{statistics.mean(col(arm, 'macro_f1_original')):>12.3f} "
                     f"{statistics.mean(col(arm, 'macro_f1_no_near_copy')):>16.3f} "
                     f"{statistics.mean(col(arm, 'test_accuracy')):>9.3f} "
                     f"{statistics.mean(col(arm, 'train_accuracy')):>9.3f} "
                     f"{hits / total:>6.3f} ({hits}/{total}) "
                     f"{statistics.mean(col(arm, 'train_seconds')):>8.0f}")
    d = [b - a for a, b in zip(col(1, "macro_f1"), col(2, "macro_f1"))]
    lines.append(f"min_ngram 2 - 1, macro F1 per fold: mean {statistics.mean(d):+.4f}, "
                 f"min {min(d):+.3f}, max {max(d):+.3f}, 2 ahead in {sum(x > 0 for x in d)}/9")
    for key in ("macro_f1_original", "macro_f1_no_near_copy"):
        lines.append(f"  {key}: 2 - 1 = {statistics.mean(col(2, key)) - statistics.mean(col(1, key)):+.4f}")
    keep_two = summary[2][0] >= summary[1][0] - 0.01 and summary[2][1] >= summary[1][1]
    lines.append(f"pre-registered rule: choose min_ngram {'2' if keep_two else '1'}")

    lines.append("\nchar_wb vocabulary size on all examples (max_ngram 4):")
    for low in (1, 2, 3, 4):
        size = len(CountVectorizer(analyzer="char_wb", ngram_range=(low, 4)).fit(texts).vocabulary_)
        lines.append(f"  min_ngram {low}: {size}")
    text = "\n".join(lines)
    (out_dir / "min_ngram_summary.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else EVAL / "cv_min_ngram_isometric-rower"
    sys.exit(main(out))
