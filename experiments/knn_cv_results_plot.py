import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.knn import KNN
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn

# Borrow the stratified K-fold CV function from main.py
def stratified_kfold_indices(y, n_folds=5, seed=0):
    """
    Yields (train_idx, valid_idx) tuples for stratified k‑fold CV.
    Does not shuffle input arrays in‑place.
    """
    rng = np.random.default_rng(seed)
    # group indices by class label
    labels = np.unique(y)
    per_class_idx = {lbl: np.where(y == lbl)[0] for lbl in labels}
    for lbl in labels:
        rng.shuffle(per_class_idx[lbl])        # shuffle in each class bucket

    folds = [list() for _ in range(n_folds)]
    for lbl, idxs in per_class_idx.items():
        for i, idx in enumerate(idxs):
            folds[i % n_folds].append(idx)

    for k in range(n_folds):
        valid_idx = np.array(folds[k])
        train_idx = np.array([idx
                              for fold in folds if fold is not folds[k]
                              for idx in fold])
        yield train_idx, valid_idx

def run_cv_multi_metric(x, y, build_model_fn, metrics_dict, param_grid, n_folds, seed=0):
    """
    Generic CV loop with multiple metrics.
    - build_model_fn(param)  → returns a *fresh* un‑trained model
    - metrics_dict: dictionary of metric functions {name: metric_fn}
    - param_grid: iterable of hyper‑parameters to try
    Returns best_param, cv_results_dict with multiple metrics
    """
    # Initialize results dict for all metrics
    cv_results = {metric_name: {p: [] for p in param_grid} for metric_name in metrics_dict}
    
    for train_idx, valid_idx in stratified_kfold_indices(y, n_folds, seed):
        xtr, xval = x[train_idx], x[valid_idx]
        ytr, yval = y[train_idx], y[valid_idx]

        for p in param_grid:
            model = build_model_fn(p)
            model.fit(xtr, ytr)
            preds = model.predict(xval)
            
            # Calculate all metrics
            for metric_name, metric_fn in metrics_dict.items():
                score = metric_fn(preds, yval)
                cv_results[metric_name][p].append(score)

    # Aggregate results for each metric
    results_aggregated = {}
    primary_metric = list(metrics_dict.keys())[0]  # Use first metric as primary
    
    for metric_name in metrics_dict:
        results_aggregated[metric_name] = {
            p: (np.mean(scores), np.std(scores)) 
            for p, scores in cv_results[metric_name].items()
        }
    
    # Get best param based on primary metric means
    avg_scores = {p: np.mean(cv_results[primary_metric][p]) for p in param_grid}
    best_param = max(avg_scores, key=avg_scores.get)   # larger = better
    
    return best_param, results_aggregated

def preprocess_data(xtrain, xtest, method):
    """Preprocess data based on the method being used"""
    # Normalize data for KNN and KMeans
    if method == "knn" or method == "kmeans":
        # needed data for normalization
        means = np.mean(xtrain, axis=0)
        stds = np.std(xtrain, axis=0)

        # apply normalization
        xtrain = normalize_fn(xtrain, means, stds)
        xtest = normalize_fn(xtest, means, stds)

        # clip values to remove extreme outliers
        med = np.median(xtrain, axis=0)
        iqr = np.percentile(xtrain, 75, axis=0) - np.percentile(xtrain, 25, axis=0) + 1e-8
        xtrain = (xtrain - med) / iqr
        xtest = (xtest - med) / iqr
        
        # apply minmax scaling
        min_vals = xtrain.min(axis=0)
        max_vals = xtrain.max(axis=0)
        xtrain = (xtrain - min_vals) / (max_vals - min_vals + 1e-8)
        xtest = (xtest - min_vals) / (max_vals - min_vals + 1e-8)

        # Handle features based on method
        if method == "kmeans":
            variances = np.var(xtrain, axis=0)
            top_k_features = np.argsort(variances)[-7:]  # Use top 7 features for KMeans
            xtrain = xtrain[:, top_k_features]
            xtest = xtest[:, top_k_features]
        else:
            # for knn, just use top 10 variance features
            variances = np.var(xtrain, axis=0)
            top_k_features = np.argsort(variances)[-10:]
            xtrain = xtrain[:, top_k_features]
            xtest = xtest[:, top_k_features]
            
    return xtrain, xtest

def plot_multi_metric_cv_results(cv_results, method_name, primary_metric_name, n_folds):
    """Plot cross-validation results with error bars for multiple metrics"""
    # Extract parameters and metric results
    params = list(cv_results[primary_metric_name].keys())
    metrics = list(cv_results.keys())
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 6), sharey=False)
    
    # Ensure axes is always an array
    if len(metrics) == 1:
        axes = [axes]
    
    # Find best parameter based on primary metric
    primary_means = [cv_results[primary_metric_name][p][0] for p in params]
    best_param = params[np.argmax(primary_means)]
    
    for i, metric_name in enumerate(metrics):
        means = [cv_results[metric_name][p][0] for p in params]
        stds = [cv_results[metric_name][p][1] for p in params]
        
        # Plot this metric
        axes[i].errorbar(params, means, yerr=stds, fmt='o-', capsize=5, markersize=8)
        
        # Mark best parameter with vertical line on all plots
        axes[i].axvline(x=best_param, color='r', linestyle='--', 
                       label=f'Best K={best_param}')
        
        # Add value annotations
        for j, (param, mean, std) in enumerate(zip(params, means, stds)):
            axes[i].annotate(f'{mean:.4f}±{std:.4f}', 
                           (param, mean),
                           textcoords="offset points",
                           xytext=(0,10), 
                           ha='center')
        
        # Set subplot specific elements
        axes[i].set_xlabel('K Value (Hyperparameter)')
        axes[i].set_ylabel(f'{metric_name} Score')
        axes[i].set_title(f'{metric_name} for {method_name.upper()}')
        axes[i].grid(True)
        axes[i].legend()
        axes[i].set_xticks(params)
    
    plt.suptitle(f'Cross-Validation Results for {method_name.upper()} ({n_folds}-Fold)')
    plt.tight_layout()
    plt.savefig(f'{method_name}_multi_metric_cv_results.png')
    plt.close()
    
    return best_param

def main(args):
    # Load data
    feature_data = np.load(args.data_path, allow_pickle=True)
    xtrain, xtest = feature_data["xtrain"], feature_data["xtest"]
    ytrain, ytest = feature_data["ytrain"], feature_data["ytest"]
    
    # Preprocess data based on the method
    xtrain, xtest = preprocess_data(xtrain, xtest, args.method)
    
    # Define parameter grid, model, and metrics based on method
    if args.method == "knn":
        param_grid = [1, 3, 5, 7, 11]
        build = lambda k: KNN(k=k)
        metrics = {
            "Accuracy": accuracy_fn,    # Primary metric
            "Macro F1": macrof1_fn      # Secondary metric
        }
        primary_metric = "Accuracy"
    elif args.method == "kmeans":
        param_grid = [3, 5, 7, 10]
        build = lambda k: KMeans(k)
        metrics = {
            "Accuracy": accuracy_fn,                 # Primary metric
            "Negative MSE": lambda *_: -mse_fn(*_)   # Secondary metric
        }
        primary_metric = "Accuracy"
    else:
        param_grid = [None]
        build = lambda _: DummyClassifier()
        metrics = {"Accuracy": accuracy_fn}
        primary_metric = "Accuracy"
    
    # Run cross-validation with multiple metrics
    best_param, cv_results = run_cv_multi_metric(
        xtrain, ytrain, build, metrics,
        param_grid, args.cv_folds, seed=42
    )
    
    # Print results for all metrics
    print(f"\nCV results ({args.cv_folds}-fold):")
    for metric_name, metric_results in cv_results.items():
        print(f"\n{metric_name} metric:")
        for p, (mean_, std_) in metric_results.items():
            print(f"  param={p:<4}: {mean_:6.3f} ± {std_:5.3f}")
    
    # Plot results for all metrics
    plot_multi_metric_cv_results(cv_results, args.method, primary_metric, args.cv_folds)
    
    print(f"\n→ Best {args.method} hyperparameter K = {best_param} (based on {primary_metric})")
    print(f"→ Results plot saved as {args.method}_multi_metric_cv_results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="knn", type=str,
                       help="Method to evaluate: knn or kmeans")
    parser.add_argument("--data_path", default="features.npz", type=str,
                       help="Path to feature data")
    parser.add_argument("--cv_folds", type=int, default=5,
                       help="Number of folds for cross-validation")
    
    args = parser.parse_args()
    main(args) 