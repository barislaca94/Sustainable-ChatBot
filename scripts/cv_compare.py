"""Compare DIET with the TF-IDF baselines on identical cross-validation folds.

Why: on one 80/20 split, DIET (0.679 macro F1) and the word+char TF-IDF
baseline (0.666) differ by about one test sentence, which is not conclusive
("explain why a small metric change is not conclusive.", NLP week 1 hands-on notebook, §18). The
lecturer's remedy is cross-validation ("For a stronger solution, use
stratified cross-validation rather than selecting the best value directly
from the test set.", NLP weeks 3-5 notebook, Additional Task 3; "Cross-validation reduces
dependence on one split", NLP week 7 notebook, §17).

Protocol, identical for every model (NLP week 7 notebook, §16: "A single stratified split
is reused so every model is evaluated on exactly the same tickets."):
  - folds: RepeatedStratifiedKFold(n_splits=3, n_repeats=3,
    random_state=42) over every example in data/nlu*.yml, so each model sees
    exactly the same 9 train/test pairs. Rasa's own CV does not seed its
    folds (rasa/nlu/test.py:1479-1481), which is why the folds are built here;
  - DIET: `rasa train nlu` on each training fold with config.yml minus the
    SafetyGate, then `rasa test nlu --successes` on the test fold. `rasa test
    nlu` reverts FallbackClassifier (rasa/nlu/test.py:1304-1308), so this is DIET alone;
  - baselines: scripts/baseline_tfidf_lr.py's two pipelines, refitted inside
    each fold ("TF-IDF preprocessing is fitted separately inside each fold
    through its pipeline.", NLP week 7 notebook, §17);
  - every metric is computed here, by the same function, from per-sentence
    predictions;
  - the SafetyGate is reported separately ("separating safety evaluation from
    routing evaluation", LLM components notebook, §22): each model is scored without it, and
    again with the gate applied to its predictions, which is what the bot
    does (the gate runs last and overrides).

Usage (takes roughly 10-20 minutes, 9 DIET trainings):
    python scripts/cv_compare.py [out folder]      # default: results/cv_compare/
"""
from __future__ import annotations

import json
import logging
import statistics
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import yaml  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from baseline_tfidf_lr import build  # noqa: E402
from components import safety_gate  # noqa: E402

SEED = 42
COSTLY_INTENT = "request_human_advisor"
MODELS = ("DIET", "word", "word+char")


def load_all():
    """Every NLU training example, merged the way Rasa reads data/."""
    from rasa.shared.nlu.training_data.loading import load_data
    from rasa.shared.nlu.training_data.training_data import TrainingData

    data = TrainingData()
    for path in sorted((REPO_ROOT / "data").glob("nlu*.yml")):
        data = data.merge(load_data(str(path)))
    return data


def write_fold(data, examples, path: Path, with_resources: bool) -> None:
    from rasa.shared.nlu.training_data.training_data import TrainingData

    fold = TrainingData(
        training_examples=examples,
        entity_synonyms=data.entity_synonyms if with_resources else None,
        regex_features=data.regex_features if with_resources else None,
        lookup_tables=data.lookup_tables if with_resources else None,
    )
    path.write_text(fold.nlu_as_yaml())


def config_without_gate(path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "config.yml").read_text())
    config["pipeline"] = [c for c in config["pipeline"]
                          if not c["name"].endswith("SafetyGate")]
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def diet_predictions(train: Path, test: Path, config: Path, work: Path) -> Dict[str, str]:
    """text -> DIET's predicted intent on the test fold."""
    model_dir = work / "model"
    run = dict(cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["rasa", "train", "nlu", "-c", str(config), "--nlu", str(train),
                    "--out", str(model_dir), "--fixed-model-name", "fold"], **run)
    results = work / "results"
    subprocess.run(["rasa", "test", "nlu", "--nlu", str(test), "--model",
                    str(model_dir / "fold.tar.gz"), "--out", str(results),
                    "--successes", "--no-plot"], **run)
    predicted = {}
    for name in ("intent_successes.json", "intent_errors.json"):
        file = results / name
        if file.exists():
            for row in json.loads(file.read_text()):
                predicted[row["text"]] = (row.get("intent_prediction") or {}).get("name")
    return predicted


def scores(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    labels = sorted(set(y_true) | set(y_pred))
    costly = [p for t, p in zip(y_true, y_pred) if t == COSTLY_INTENT]
    return {
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "rha_recall": sum(p == COSTLY_INTENT for p in costly) / len(costly) if costly else 0.0,
    }


def main(out_dir: Path) -> int:
    data = load_all()
    examples = [m for m in data.training_examples if m.get("intent")]
    texts = [m.get("text") for m in examples]
    intents = [m.get("intent") for m in examples]
    categories = safety_gate.load_categories(REPO_ROOT / "data" / "safety_patterns.yml")
    gate = [safety_gate.classify(t, categories, 500) for t in texts]

    out_dir.mkdir(parents=True, exist_ok=True)
    folds = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=SEED)
    rows = []          # one row per (fold, model, gate)
    predictions = []   # per sentence, per fold, for the error analysis

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config = tmp / "config_no_gate.yml"
        config_without_gate(config)
        for k, (train_idx, test_idx) in enumerate(folds.split(texts, intents)):
            work = tmp / f"fold{k}"
            work.mkdir()
            write_fold(data, [examples[i] for i in train_idx], work / "train.yml", True)
            write_fold(data, [examples[i] for i in test_idx], work / "test.yml", False)

            y_true = [intents[i] for i in test_idx]
            preds = {}
            diet = diet_predictions(work / "train.yml", work / "test.yml", config, work)
            preds["DIET"] = [diet.get(texts[i]) or "none" for i in test_idx]
            for features in ("word", "word+char"):
                model = build(features)
                model.fit([texts[i] for i in train_idx], [intents[i] for i in train_idx])
                preds[features] = list(model.predict([texts[i] for i in test_idx]))

            for name in MODELS:
                gated = [gate[i][1] if gate[i] else p for i, p in zip(test_idx, preds[name])]
                rows.append({"fold": k, "model": name, "gate": False, **scores(y_true, preds[name])})
                rows.append({"fold": k, "model": name, "gate": True, **scores(y_true, gated)})
            for j, i in enumerate(test_idx):
                predictions.append({"fold": k, "text": texts[i], "true": intents[i],
                                    "gate": gate[i][1] if gate[i] else None,
                                    **{name: preds[name][j] for name in MODELS}})
            missing = sum(1 for i in test_idx if texts[i] not in diet)
            print(f"fold {k}: train {len(train_idx)}, test {len(test_idx)}, "
                  f"DIET predictions missing {missing}", flush=True)

    (out_dir / "cv_rows.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "cv_predictions.json").write_text(json.dumps(predictions, indent=2))

    def column(model, gated, metric):
        return [r[metric] for r in rows if r["model"] == model and r["gate"] == gated]

    lines = [f"examples {len(texts)}, intents {len(set(intents))}, "
             f"RepeatedStratifiedKFold(3 x 3, random_state={SEED}); "
             f"gate fires on {sum(1 for g in gate if g)} examples",
             "mean (std, ddof=1) over 9 folds", ""]
    for gated in (False, True):
        lines.append("WITH the SafetyGate applied to every model" if gated
                     else "Classifier only (no SafetyGate)")
        lines.append(f"{'model':10s} {'accuracy':>16s} {'macro F1':>16s} {'RHA recall':>16s}")
        for model in MODELS:
            cells = []
            for metric in ("accuracy", "macro_f1", "rha_recall"):
                v = column(model, gated, metric)
                cells.append(f"{statistics.mean(v):.3f} ({statistics.stdev(v):.3f})")
            lines.append(f"{model:10s} " + " ".join(f"{c:>16s}" for c in cells))
        for base in ("word", "word+char"):
            d = [a - b for a, b in zip(column("DIET", gated, "macro_f1"),
                                       column(base, gated, "macro_f1"))]
            wins = sum(x > 0 for x in d)
            lines.append(f"  DIET - {base:9s} macro F1: mean {statistics.mean(d):+.3f}, "
                         f"min {min(d):+.3f}, max {max(d):+.3f}, DIET ahead in {wins}/9 folds")
        lines.append("")

    # Per-intent F1 pooled over the three repeats (each repeat predicts every
    # example once), classifier only.
    lines.append("per-intent F1, pooled over all folds, classifier only:")
    lines.append(f"{'intent':24s} {'n':>4s} " + " ".join(f"{m:>9s}" for m in MODELS))
    y_all = [p["true"] for p in predictions]
    labels = sorted(set(y_all))
    per = {m: f1_score(y_all, [p[m] for p in predictions], labels=labels,
                       average=None, zero_division=0) for m in MODELS}
    for idx, intent in enumerate(labels):
        n = sum(1 for y in intents if y == intent)
        lines.append(f"{intent:24s} {n:>4d} " + " ".join(f"{per[m][idx]:>9.2f}" for m in MODELS))

    text = "\n".join(lines)
    (out_dir / "cv_summary.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "results" / "cv_compare"
    sys.exit(main(out))
