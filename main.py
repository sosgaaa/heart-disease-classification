"""Evaluate three classifiers on the heart disease feature dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.methods.kmeans import KMeans
from src.methods.knn import KNN
from src.methods.logistic_regression import LogisticRegression
from src.utils import accuracy_fn, append_bias_term, macrof1_fn

DEFAULT_DATA = Path(__file__).with_name("features.npz")


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        xtrain, xtest, ytrain, ytest = (data[key] for key in ("xtrain", "xtest", "ytrain", "ytest"))
    if xtrain.ndim != 2 or xtest.ndim != 2 or xtrain.shape[1] != xtest.shape[1]:
        raise ValueError("Training and test features must be 2D with matching columns")
    if ytrain.shape != (len(xtrain),) or ytest.shape != (len(xtest),):
        raise ValueError("Each sample must have one label")
    return xtrain, xtest, ytrain.astype(int), ytest.astype(int)


def preprocess(training: np.ndarray, evaluation: np.ndarray, method: str):
    """Fit all transforms on training rows only."""
    mean = training.mean(axis=0)
    std = np.where(training.std(axis=0) == 0, 1, training.std(axis=0))
    xtrain = (training - mean) / std
    xeval = (evaluation - mean) / std
    if method == "logistic_regression":
        return append_bias_term(xtrain), append_bias_term(xeval)
    xtrain, xeval = np.clip(xtrain, -3, 3), np.clip(xeval, -3, 3)
    minimum = xtrain.min(axis=0)
    span = xtrain.max(axis=0) - minimum
    span = np.where(span == 0, 1, span)
    xtrain, xeval = (xtrain - minimum) / span, (xeval - minimum) / span
    feature_count = min(xtrain.shape[1], 7 if method == "kmeans" else 10)
    selected = np.argsort(xtrain.var(axis=0))[-feature_count:]
    return xtrain[:, selected], xeval[:, selected]


def stratified_folds(labels: np.ndarray, count: int, seed: int):
    if count < 2:
        raise ValueError("Cross-validation needs at least two folds")
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(count)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            folds[position % count].append(index)
    for fold in folds:
        validation = np.asarray(fold, dtype=int)
        training = np.setdiff1d(np.arange(len(labels)), validation)
        yield training, validation


def build_model(args: argparse.Namespace):
    if args.method == "knn":
        return KNN(k=args.k)
    if args.method == "kmeans":
        return KMeans(k=args.k, random_state=args.seed)
    return LogisticRegression(lr=args.lr, max_iters=args.max_iters)


def evaluate(model, xtrain, ytrain, xeval, yeval):
    model.fit(xtrain, ytrain)
    predictions = model.predict(xeval)
    return accuracy_fn(predictions, yeval), macrof1_fn(predictions, yeval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--method", choices=("knn", "kmeans", "logistic_regression"), default="knn")
    parser.add_argument("--k", type=int, default=None, help="neighbors or clusters")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.k is None:
        args.k = 34 if args.method == "kmeans" else 7
    if args.k < 1:
        parser.error("--k must be positive")

    xtrain, xtest, ytrain, ytest = load_dataset(args.data)
    if args.cv_folds:
        results = []
        for train_idx, valid_idx in stratified_folds(ytrain, args.cv_folds, args.seed):
            xfit, xvalid = preprocess(xtrain[train_idx], xtrain[valid_idx], args.method)
            results.append(evaluate(build_model(args), xfit, ytrain[train_idx], xvalid, ytrain[valid_idx]))
        mean, std = np.mean(results, axis=0), np.std(results, axis=0)
        print(f"CV ({args.cv_folds} folds): accuracy {mean[0]:.2f}% ± {std[0]:.2f}; macro F1 {mean[1]:.3f} ± {std[1]:.3f}")

    xfit, xeval = preprocess(xtrain, xtest, args.method)
    accuracy, f1 = evaluate(build_model(args), xfit, ytrain, xeval, ytest)
    print(f"Test: accuracy {accuracy:.2f}%; macro F1 {f1:.3f}")


if __name__ == "__main__":
    main()
