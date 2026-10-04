"""Measure DIET at 50 / 100 / 200 epochs on identical cross-validation folds.

The protocol and the decision rule were written down before this script was run:
  - the same seeded folds as scripts/cv_compare.py (RepeatedStratifiedKFold
    3 x 3, random_state 42) over all examples in data/nlu*.yml;
  - DIET `epochs` is the only change ("Change only one variable, for example
    learning rate, epoch count, or batch size.", transformer fine-tuning notebook,
    Challenge 5); the
    SafetyGate is left out, and `rasa test nlu` reverts FallbackClassifier
    (rasa/nlu/test.py:1304-1308), so this measures DIET alone;
  - deciding metric: macro F1 over all examples, per fold; RHA recall pooled
    over all folds and compared with 100 epochs (the current config);
  - also reported: macro F1 on the original data/nlu.yml sentences, macro F1
    without near-copy test items (char_wb 3-5 cosine >= 0.80 to a
    same-intent sentence of the training fold, the threshold used when the
    training data was merged),
    training-fold accuracy (dog-breed notebook §14, overfitting indicator) and training
    time (transformer fine-tuning notebook, Challenge 1).

Usage (about 45 minutes, 27 DIET trainings):
    python scripts/cv_epochs.py [out folder]
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import yaml  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from cv_compare import load_all, write_fold  # noqa: E402

SEED = 42
EPOCHS = (50, 100, 200)
REFERENCE_EPOCHS = 100
COSTLY_INTENT = "request_human_advisor"
NEAR_COPY = 0.80
RUN = dict(cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def write_config(path: Path, epochs: int) -> None:
    """config.yml without the SafetyGate and with DIET's epochs set."""
    config = yaml.safe_load((REPO_ROOT / "config.yml").read_text())
    config["pipeline"] = [c for c in config["pipeline"]
                          if not c["name"].endswith("SafetyGate")]
    for component in config["pipeline"]:
        if component["name"] == "DIETClassifier":
            component["epochs"] = epochs
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def predict(model: Path, data: Path, out: Path) -> Dict[str, str]:
    """text -> predicted intent, from `rasa test nlu --successes`."""
    subprocess.run(["rasa", "test", "nlu", "--nlu", str(data), "--model", str(model),
                    "--out", str(out), "--successes", "--no-plot"], **RUN)
    predicted = {}
    for name in ("intent_successes.json", "intent_errors.json"):
        file = out / name
        if file.exists():
            for row in json.loads(file.read_text()):
                predicted[row["text"]] = (row.get("intent_prediction") or {}).get("name")
    return predicted


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


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
    # Near-copy test items per fold: nearest training-fold sentence of the
    # same intent at char_wb 3-5 cosine >= 0.80.
    near_copy = []
    for train_idx, test_idx in folds:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit([texts[i] for i in train_idx])
        sim = cosine_similarity(vec.transform([texts[i] for i in test_idx]),
                                vec.transform([texts[i] for i in train_idx]))
        flags = set()
        for r, i in enumerate(test_idx):
            j = sim[r].argmax()
            if sim[r, j] >= NEAR_COPY and intents[train_idx[j]] == intents[i]:
                flags.add(i)
        near_copy.append(flags)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows, predictions = [], []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k, (train_idx, test_idx) in enumerate(folds):
            fold = tmp / f"fold{k}"
            fold.mkdir()
            write_fold(data, [examples[i] for i in train_idx], fold / "train.yml", True)
            write_fold(data, [examples[i] for i in test_idx], fold / "test.yml", False)
            write_fold(data, [examples[i] for i in train_idx], fold / "train_eval.yml", False)
            for epochs in EPOCHS:
                work = fold / f"e{epochs}"
                work.mkdir()
                write_config(work / "config.yml", epochs)
                # A fresh, empty training cache for every run: Rasa reuses cached
                # components (rasa/engine/caching.py:34, RASA_CACHE_DIRECTORY), and a
                # cache hit would skip DIET training and make the timing meaningless.
                env = {**os.environ, "RASA_CACHE_DIRECTORY": str(work / "cache")}
                started = time.time()
                subprocess.run(["rasa", "train", "nlu", "-c", str(work / "config.yml"),
                                "--nlu", str(fold / "train.yml"), "--out", str(work),
                                "--fixed-model-name", "model"], env=env, **RUN)
                seconds = time.time() - started
                model = work / "model.tar.gz"
                test_pred = predict(model, fold / "test.yml", work / "test")
                train_pred = predict(model, fold / "train_eval.yml", work / "train")
                missing = [i for i in test_idx if texts[i] not in test_pred] + \
                          [i for i in train_idx if texts[i] not in train_pred]
                if missing:
                    raise SystemExit(f"fold {k}, {epochs} epochs: {len(missing)} predictions missing")

                y_true = [intents[i] for i in test_idx]
                y_pred = [test_pred[texts[i]] for i in test_idx]
                keep_orig = [j for j, i in enumerate(test_idx) if texts[i] in original]
                keep_clean = [j for j, i in enumerate(test_idx) if i not in near_copy[k]]
                rows.append({
                    "fold": k, "epochs": epochs, "train_seconds": round(seconds, 1),
                    "macro_f1": macro_f1(y_true, y_pred),
                    "macro_f1_original": macro_f1([y_true[j] for j in keep_orig],
                                                  [y_pred[j] for j in keep_orig]),
                    "macro_f1_no_near_copy": macro_f1([y_true[j] for j in keep_clean],
                                                      [y_pred[j] for j in keep_clean]),
                    "test_accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true),
                    "train_accuracy": sum(train_pred[texts[i]] == intents[i] for i in train_idx)
                                      / len(train_idx),
                    "rha_hits": sum(1 for t, p in zip(y_true, y_pred) if t == COSTLY_INTENT and p == t),
                    "rha_total": sum(1 for t in y_true if t == COSTLY_INTENT),
                })
                predictions += [{"fold": k, "epochs": epochs, "text": texts[i],
                                 "true": intents[i], "pred": test_pred[texts[i]]} for i in test_idx]
                print(f"fold {k} epochs {epochs}: {seconds:.0f}s, "
                      f"near-copy test items excluded {len(near_copy[k])}", flush=True)
                (out_dir / "epochs_rows.json").write_text(json.dumps(rows, indent=2))

    (out_dir / "epochs_predictions.json").write_text(json.dumps(predictions, indent=2))

    def col(epochs, key):
        return [r[key] for r in rows if r["epochs"] == epochs]

    lines = [f"examples {len(texts)}; RepeatedStratifiedKFold(3 x 3, random_state={SEED}); "
             "DIET only (no SafetyGate, fallback reverted); mean (std, ddof=1) over 9 folds", ""]
    head = (f"{'epochs':>6s} {'macro F1':>15s} {'F1 original':>15s} {'F1 no near-copy':>16s} "
            f"{'test acc':>9s} {'train acc':>9s} {'RHA recall':>14s} {'train s':>8s}")
    lines.append(head)
    summary = {}
    for epochs in EPOCHS:
        hits, total = sum(col(epochs, "rha_hits")), sum(col(epochs, "rha_total"))
        f1 = col(epochs, "macro_f1")
        summary[epochs] = {"macro_f1": statistics.mean(f1), "rha": hits / total}
        cells = [f"{statistics.mean(f1):.3f} ({statistics.stdev(f1):.3f})"]
        for key in ("macro_f1_original", "macro_f1_no_near_copy"):
            v = col(epochs, key)
            cells.append(f"{statistics.mean(v):.3f} ({statistics.stdev(v):.3f})")
        lines.append(f"{epochs:>6d} {cells[0]:>15s} {cells[1]:>15s} {cells[2]:>16s} "
                     f"{statistics.mean(col(epochs, 'test_accuracy')):>9.3f} "
                     f"{statistics.mean(col(epochs, 'train_accuracy')):>9.3f} "
                     f"{hits / total:>6.3f} ({hits}/{total}) "
                     f"{statistics.mean(col(epochs, 'train_seconds')):>8.0f}")
    lines.append("")
    for a, b in ((50, 100), (200, 100)):
        d = [x - y for x, y in zip(col(a, "macro_f1"), col(b, "macro_f1"))]
        lines.append(f"{a} - {b} epochs, macro F1 per fold: mean {statistics.mean(d):+.3f}, "
                     f"min {min(d):+.3f}, max {max(d):+.3f}, {a} ahead in {sum(x > 0 for x in d)}/9")

    best = max(summary.values(), key=lambda s: s["macro_f1"])["macro_f1"]
    reference_rha = summary[REFERENCE_EPOCHS]["rha"]
    eligible = [e for e in EPOCHS if summary[e]["macro_f1"] >= best - 0.01
                and summary[e]["rha"] >= reference_rha]
    lines += ["", f"pre-registered rule: best macro F1 {best:.3f}; eligible (within 0.01 and RHA recall "
              f">= {reference_rha:.3f} at {REFERENCE_EPOCHS} epochs): {eligible}; "
              f"smallest: {min(eligible) if eligible else 'none - review the decision'}"]
    text = "\n".join(lines)
    (out_dir / "epochs_summary.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO_ROOT / "evaluation" / "phase7_2026-10-03" / "cv_epochs_isometric-rower")
    sys.exit(main(out))
