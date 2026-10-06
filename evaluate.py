"""Compare three classifiers with nested CV and training-only parameter selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from main import DEFAULT_DATA, load_dataset, preprocess, stratified_folds
from src.methods.kmeans import KMeans
from src.methods.knn import KNN
from src.methods.logistic_regression import LogisticRegression

GRIDS = {
    "knn": [{"k": k, "features": f} for f in (7, 10, 13) for k in (1, 3, 5, 7, 9, 11, 15)],
    "kmeans": [{"k": k, "features": f} for f in (7, 10, 13) for k in (5, 10, 20, 30, 34, 40)],
    "logistic_regression": [
        {"lr": lr, "max_iters": iters, "reg": 0.001}
        for lr in (0.001, 0.01, 0.1) for iters in (250, 500, 1000)
    ],
}
METHOD_NAMES = {"baseline": "Majority baseline", "knn": "KNN", "kmeans": "K-Means", "logistic_regression": "Logistic regression"}


def metrics(truth, predictions, classes):
    """Use the same fixed class set for every fold, including zero-recall classes."""
    confusion = np.array([
        [np.sum((truth == actual) & (predictions == predicted)) for predicted in classes]
        for actual in classes
    ], dtype=int)
    tp = confusion.diagonal()
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(len(classes)), where=denominator != 0)
    support = confusion.sum(axis=1)
    recall = np.divide(tp, support, out=np.zeros(len(classes)), where=support != 0)
    return {
        "accuracy": float(np.mean(truth == predictions) * 100),
        "macro_f1": float(f1.mean()),
        "class_f1": f1.tolist(), "class_recall": recall.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def summarize(folds):
    return {key: {"mean": float(np.mean([fold[key] for fold in folds])),
                  "std": float(np.std([fold[key] for fold in folds], ddof=0))}
            for key in ("accuracy", "macro_f1")}


def predict(method, params, training, labels, evaluation, seed):
    if method == "baseline":
        values, counts = np.unique(labels, return_counts=True)
        return np.full(len(evaluation), values[np.argmax(counts)])
    xfit, xeval = preprocess(training, evaluation, method, params.get("features"))
    if method == "knn":
        model = KNN(k=params["k"])
    elif method == "kmeans":
        model = KMeans(k=params["k"], random_state=seed)
    else:
        model = LogisticRegression(**params)
    model.fit(xfit, labels)
    return model.predict(xeval)


def select_parameters(method, training, labels, candidates, fold_count, seed):
    """Selection accepts training data only; no test samples or labels enter this API."""
    classes = np.unique(labels)
    folds = list(stratified_folds(labels, fold_count, seed))
    search = []
    for params in candidates:
        scores = [metrics(labels[valid], predict(method, params, training[fit], labels[fit], training[valid], seed), classes)
                  for fit, valid in folds]
        search.append({"params": dict(params), "cv": summarize(scores)})
    # Stable ties: highest macro F1, then accuracy, then first declared candidate.
    best = max(search, key=lambda row: (row["cv"]["macro_f1"]["mean"], row["cv"]["accuracy"]["mean"]))
    return dict(best["params"]), search


def study_method(method, training, labels, candidates, outer_folds, inner_folds, seed):
    classes = np.unique(labels)
    oof_predictions = np.empty_like(labels)
    records = []
    for number, (fit, valid) in enumerate(stratified_folds(labels, outer_folds, seed), 1):
        params, _ = select_parameters(method, training[fit], labels[fit], candidates, inner_folds, seed)
        predictions = predict(method, params, training[fit], labels[fit], training[valid], seed)
        oof_predictions[valid] = predictions
        records.append({"fold": number, "params": params, "validation_indices": valid.tolist(),
                        **metrics(labels[valid], predictions, classes)})
    params, search = select_parameters(method, training, labels, candidates, outer_folds, seed)
    return {"selected_params": params, "search": search, "nested_folds": records,
            "nested_cv": summarize(records), "out_of_fold": metrics(labels, oof_predictions, classes)}


def run_study(training, labels, evaluation, evaluation_labels, grids=None, outer_folds=5, inner_folds=4, seed=42):
    grids = GRIDS if grids is None else grids
    classes = np.unique(labels)
    result = {"config": {"seed": seed, "outer_folds": outer_folds, "inner_folds": inner_folds,
                         "selection_metric": "mean macro F1; accuracy breaks ties", "grids": grids},
              "data": {"classes": classes.tolist(), "train_shape": list(training.shape),
                       "test_shape": list(evaluation.shape),
                       "train_class_counts": [int(np.sum(labels == c)) for c in classes],
                       "test_class_counts": [int(np.sum(evaluation_labels == c)) for c in classes]},
              "test_status": "Used in exploratory parameter searches; fixed-test scores are descriptive.",
              "methods": {}}
    # Complete every selection before evaluating any model on the fixed test split.
    for method, candidates in grids.items():
        print(f"Nested CV and training-only selection: {METHOD_NAMES[method]}", flush=True)
        result["methods"][method] = study_method(method, training, labels, candidates, outer_folds, inner_folds, seed)
    baseline_folds = [metrics(labels[valid], predict("baseline", {}, training[fit], labels[fit], training[valid], seed), classes)
                      for fit, valid in stratified_folds(labels, outer_folds, seed)]
    result["methods"]["baseline"] = {"selected_params": {}, "nested_cv": summarize(baseline_folds)}
    for method, record in result["methods"].items():
        predictions = predict(method, record["selected_params"], training, labels, evaluation, seed)
        record["test"] = metrics(evaluation_labels, predictions, classes)
        record["test"]["predictions"] = predictions.tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/evaluation.json")
    args = parser.parse_args()
    training, evaluation, labels, evaluation_labels = load_dataset(args.data)
    result = run_study(training, labels, evaluation, evaluation_labels)
    result["data"]["sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    result["provenance"] = {
        "numpy_version": np.__version__,
        "source_sha256": {str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__), Path(__file__).with_name("main.py"),
                                       *sorted((Path(__file__).parent / "src").rglob("*.py"))]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for method in ("baseline", *GRIDS):
        record = result["methods"][method]
        print(f"{METHOD_NAMES[method]}: nested F1 {record['nested_cv']['macro_f1']['mean']:.3f}; "
              f"descriptive test accuracy {record['test']['accuracy']:.2f}%, F1 {record['test']['macro_f1']:.3f}; "
              f"parameters {record['selected_params']}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
