import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import time # To estimate time for the K sweep

# Assuming src package is accessible
from src.methods.knn import KNN
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, accuracy_fn, macrof1_fn

np.random.seed(100) # Consistent seed

# --- Helper functions copied from main.py (stratified_kfold_indices, run_cv) ---
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

def run_cv(x, y, build_model_fn, metric_fn, param_grid, n_folds, seed=0):
    """
    Generic CV loop.
    - build_model_fn(param)  → returns a *fresh* un‑trained model
    - metric_fn(y_pred, y_true) → returns scalar
    - param_grid: iterable of hyper‑parameters to try
    Returns best_param, cv_results_dict {param: (mean_score, std_score)}
    """
    cv_results = {p: [] for p in param_grid}

    for train_idx, valid_idx in stratified_kfold_indices(y, n_folds, seed):
        xtr, xval = x[train_idx], x[valid_idx]
        ytr, yval = y[train_idx], y[valid_idx]

        for p in param_grid:
            model = build_model_fn(p)
            model.fit(xtr, ytr)
            preds = model.predict(xval)
            cv_results[p].append(metric_fn(preds, yval))

    # aggregate
    avg_scores = {p: np.mean(scores) for p, scores in cv_results.items()}
    best_param = max(avg_scores, key=avg_scores.get)   # larger = better
    return best_param, {p: (np.mean(s), np.std(s)) for p, s in cv_results.items()}

# --- Preprocessing Function ---

def preprocess_and_select_features(xtrain, xtest, method_name):
    """
    Applies preprocessing (Z-score -> Median/IQR -> MinMax) and feature selection
    based on the method, mirroring main.py's logic for the --test flag or CV.
    Fits preprocessing on xtrain and applies to both xtrain and xtest.
    """
    print(f"Preprocessing and selecting features for {method_name}...")
    xtrain_proc = xtrain.copy()
    xtest_proc = xtest.copy()

    # 1. Z-score Normalization
    means = np.mean(xtrain_proc, axis=0)
    stds = np.std(xtrain_proc, axis=0)
    stds[stds == 0] = 1e-8 # Avoid division by zero
    xtrain_proc = normalize_fn(xtrain_proc, means, stds)
    xtest_proc = normalize_fn(xtest_proc, means, stds)

    # 2. Median/IQR Scaling
    med = np.median(xtrain_proc, axis=0)
    iqr = np.percentile(xtrain_proc, 75, axis=0) - np.percentile(xtrain_proc, 25, axis=0) + 1e-8
    xtrain_proc = (xtrain_proc - med) / iqr
    xtest_proc = (xtest_proc - med) / iqr

    # 3. MinMax Scaling
    min_vals = xtrain_proc.min(axis=0)
    max_vals = xtrain_proc.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-8
    xtrain_proc = (xtrain_proc - min_vals) / range_vals
    xtest_proc = (xtest_proc - min_vals) / range_vals

    # 4. Feature Selection (based on variance of preprocessed training data)
    variances = np.var(xtrain_proc, axis=0)
    top_k_features_indices = slice(None) # Default to all features

    if method_name == "kmeans":
        n_features_kmeans = 7 # As used in main.py for kmeans test/validation mode
        if xtrain_proc.shape[1] >= n_features_kmeans:
            top_k_features_indices = np.argsort(variances)[-n_features_kmeans:]
            print(f"  Selected top {n_features_kmeans} features for KMeans based on variance.")
        else:
             print(f"  Warning: Not enough features ({xtrain_proc.shape[1]}) for KMeans selection (needs {n_features_kmeans}). Using all.")
    elif method_name == "knn":
        n_features_knn = 10 # As used in main.py for knn test/validation mode/CV
        if xtrain_proc.shape[1] >= n_features_knn:
            top_k_features_indices = np.argsort(variances)[-n_features_knn:]
            print(f"  Selected top {n_features_knn} features for KNN based on variance.")
        else:
            print(f"  Warning: Not enough features ({xtrain_proc.shape[1]}) for KNN selection (needs {n_features_knn}). Using all.")

    xtrain_final = xtrain_proc[:, top_k_features_indices]
    xtest_final = xtest_proc[:, top_k_features_indices]
    print(f"  Final feature shape: Train={xtrain_final.shape}, Test={xtest_final.shape}")

    return xtrain_final, xtest_final

# --- Plotting Functions ---

# plot_knn_cv remains the same
def plot_knn_cv(xtrain_full, ytrain_full, n_folds, param_grid, filename="knn_cv_accuracy_plot.png"):
    """
    Performs KNN cross-validation and plots the results.
    """
    print("\n--- Generating KNN Cross-Validation Plot ---")
    if n_folds <= 1:
        print("  Skipping CV plot: n_folds must be > 1.")
        return

    # Preprocess data specifically for KNN CV (using top 10 features)
    xtrain_knn_cv, _ = preprocess_and_select_features(xtrain_full, xtrain_full, "knn") # Use dummy test set

    build = lambda k: KNN(k=k)
    metric = accuracy_fn
    best_param, cv_table = run_cv(
        xtrain_knn_cv, ytrain_full, build, metric,
        param_grid, n_folds, seed=42
    )

    params = list(cv_table.keys())
    means = [cv_table[p][0] for p in params]
    stds = [cv_table[p][1] for p in params]

    plt.figure(figsize=(10, 6))
    plt.errorbar(params, means, yerr=stds, fmt='o-', capsize=5, markersize=8, label='Mean Accuracy ± Std Dev')
    plt.axvline(x=best_param, color='r', linestyle='--', label=f'Best K={best_param} (Acc={cv_table[best_param][0]:.2f}%)')

    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Cross-Validation Accuracy (%)')
    plt.title(f'KNN {n_folds}-Fold Cross-Validation Accuracy')
    # Adjust x-ticks if param_grid is large
    if len(params) > 20:
        plt.xticks(params[::len(params)//10]) # Show ~10 ticks
    else:
        plt.xticks(params)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"  CV plot saved as {filename}")
    plt.close()

# plot_validation_splits remains the same
def plot_validation_splits(xtrain_orig, ytrain_orig, method_name, method_args,
                           validation_percentages, filename="validation_split_accuracy_plot.png"):
    """
    Evaluates a method over different validation split percentages using a FIXED K
    and plots accuracy.
    """
    print(f"\n--- Generating Validation Split Plot for {method_name.upper()} (Fixed K={method_args.get('K', 'Default')}) ---")
    results_per_split = {}

    for p in validation_percentages:
        print(f"  Testing Validation Split: {p*100:.0f}%")

        # Split original training data
        num_train_samples = xtrain_orig.shape[0]
        validation_size = int(num_train_samples * p)
        if validation_size == 0 or num_train_samples - validation_size == 0:
            print(f"  Warning: Split size invalid for {p*100:.0f}%. Skipping.")
            continue

        shuffled_indices = np.random.permutation(num_train_samples)
        valid_indices = shuffled_indices[:validation_size]
        train_indices = shuffled_indices[validation_size:]

        xtrain_split = xtrain_orig[train_indices]
        ytrain_split = ytrain_orig[train_indices]
        xvalid_split = xtrain_orig[valid_indices]
        yvalid_split = ytrain_orig[valid_indices]

        # Preprocess *within the loop* based on the current training split
        xtrain_split_proc, xvalid_split_proc = preprocess_and_select_features(
            xtrain_split, xvalid_split, method_name
        )

        # Initialize Model with fixed K
        if method_name == "knn":
            model = KNN(k=method_args.get('K', 1)) # Use provided K or default
        elif method_name == "kmeans":
             num_clusters = method_args.get('K', len(np.unique(ytrain_split)))
             model = KMeans(k=num_clusters)
        else:
            print(f"  Skipping plot: Method {method_name} not supported.")
            return

        # Train and Evaluate
        try:
            model.fit(xtrain_split_proc, ytrain_split)
            preds_valid = model.predict(xvalid_split_proc)
            acc_valid = accuracy_fn(preds_valid, yvalid_split)
            results_per_split[p] = acc_valid
            print(f"    Validation Accuracy: {acc_valid:.3f}%")
        except Exception as e:
            print(f"    Error during model fit/predict for split {p*100:.0f}%: {e}")
            results_per_split[p] = -1 # Indicate error

    # Plotting
    valid_splits = [p for p, acc in results_per_split.items() if acc >= 0]
    valid_accs = [results_per_split[p] for p in valid_splits]

    if not valid_splits:
        print("  No valid results to plot for validation splits.")
        return

    best_split_idx = np.argmax(valid_accs)
    best_split = valid_splits[best_split_idx]
    best_acc = valid_accs[best_split_idx]

    plt.figure(figsize=(10, 6))
    plt.plot([p * 100 for p in valid_splits], valid_accs, 'o-', label=f'{method_name.upper()} Validation Accuracy (K={method_args.get("K", "Default")})')
    plt.axvline(x=best_split * 100, color='r', linestyle='--', label=f'Best Split={best_split*100:.0f}% (Acc={best_acc:.2f}%)')

    plt.xlabel('Validation Set Size (%)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{method_name.upper()} Accuracy vs. Validation Split Percentage (Fixed K)')
    plt.xticks([p * 100 for p in validation_percentages])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"  Validation split plot saved as {filename}")
    plt.close()

# --- NEW Plotting Function for Best K Sweep ---
def plot_best_k_validation_splits(xtrain_orig, ytrain_orig, method_name, k_range,
                                  validation_percentages, filename="best_k_validation_split_plot.png"):
    """
    Evaluates a method over different validation splits, finding the BEST K within k_range
    for each split, and plots the resulting best accuracy.
    """
    print(f"\n--- Generating Best K Validation Split Plot for {method_name.upper()} (K={k_range[0]}-{k_range[-1]}) ---")
    best_results_per_split = {} # Stores {split_percentage: (best_accuracy, best_k)}

    start_time_total = time.time()

    # Outer loop: Validation Split Percentage
    for p in validation_percentages:
        print(f"  Testing Validation Split: {p*100:.0f}%")
        start_time_split = time.time()

        # Split original training data
        num_train_samples = xtrain_orig.shape[0]
        validation_size = int(num_train_samples * p)
        if validation_size == 0 or num_train_samples - validation_size == 0:
            print(f"  Warning: Split size invalid for {p*100:.0f}%. Skipping.")
            continue

        shuffled_indices = np.random.permutation(num_train_samples)
        valid_indices = shuffled_indices[:validation_size]
        train_indices = shuffled_indices[validation_size:]

        xtrain_split = xtrain_orig[train_indices]
        ytrain_split = ytrain_orig[train_indices]
        xvalid_split = xtrain_orig[valid_indices]
        yvalid_split = ytrain_orig[valid_indices]

        # Preprocess *once* for this split, before the K loop
        xtrain_split_proc, xvalid_split_proc = preprocess_and_select_features(
            xtrain_split, xvalid_split, method_name
        )

        current_best_acc_for_split = -1.0
        current_best_k_for_split = -1

        # Inner loop: Hyperparameter K
        for k_val in k_range:
            # Initialize Model with current K
            if method_name == "knn":
                model = KNN(k=k_val)
            elif method_name == "kmeans":
                 # For KMeans, k_val is the number of clusters
                 # Use random_state for reproducibility if KMeans supports it
                 model = KMeans(k=k_val, random_state=42) # Assuming KMeans has random_state
            else:
                print(f"  Skipping plot: Method {method_name} not supported.")
                return

            # Train and Evaluate
            try:
                model.fit(xtrain_split_proc, ytrain_split)
                preds_valid = model.predict(xvalid_split_proc)
                acc_valid = accuracy_fn(preds_valid, yvalid_split)

                if acc_valid > current_best_acc_for_split:
                    current_best_acc_for_split = acc_valid
                    current_best_k_for_split = k_val

                # Optional: Print progress within K loop
                # if k_val % 10 == 0: print(f"    ... K={k_val}, Acc={acc_valid:.2f}%")

            except Exception as e:
                print(f"    Error during model fit/predict for split {p*100:.0f}%, K={k_val}: {e}")
                # Continue to next K, don't update best accuracy for this K

        # Store best result for this split percentage
        best_results_per_split[p] = (current_best_acc_for_split, current_best_k_for_split)
        split_duration = time.time() - start_time_split
        print(f"    => Best Accuracy for split {p*100:.0f}%: {current_best_acc_for_split:.3f}% (found with K={current_best_k_for_split}) - Took {split_duration:.1f}s")


    # Plotting
    valid_splits = [p for p, (acc, k) in best_results_per_split.items() if acc >= 0]
    best_accs = [best_results_per_split[p][0] for p in valid_splits]
    best_ks = [best_results_per_split[p][1] for p in valid_splits]

    if not valid_splits:
        print("  No valid results to plot for best K validation splits.")
        return

    overall_best_split_idx = np.argmax(best_accs)
    overall_best_split = valid_splits[overall_best_split_idx]
    overall_best_acc = best_accs[overall_best_split_idx]
    k_at_overall_best = best_ks[overall_best_split_idx]

    plt.figure(figsize=(10, 6))
    plt.plot([p * 100 for p in valid_splits], best_accs, 'o-', label=f'Best {method_name.upper()} Validation Accuracy (K optimized)')

    # Annotate points with the best K found
    for i, p in enumerate(valid_splits):
        plt.annotate(f"K={best_ks[i]}", ([p * 100 for p in valid_splits][i], best_accs[i]), textcoords="offset points", xytext=(0,5), ha='center')

    plt.axvline(x=overall_best_split * 100, color='r', linestyle='--', label=f'Overall Best: Split={overall_best_split*100:.0f}% \n(Acc={overall_best_acc:.2f}%, K={k_at_overall_best})')

    plt.xlabel('Validation Set Size (%)')
    plt.ylabel('Best Accuracy (%)')
    plt.title(f'Best {method_name.upper()} Accuracy vs. Validation Split (Optimized K in {k_range[0]}-{k_range[-1]})')
    plt.xticks([p * 100 for p in validation_percentages])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"  Best K validation split plot saved as {filename}")
    plt.close()

    total_duration = time.time() - start_time_total
    print(f"  Total time for Best K sweep plot: {total_duration:.1f}s")


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots based on main.py evaluation logic.")
    parser.add_argument(
        "--method", default="knn", type=str,
        help="Method to evaluate: knn / kmeans"
    )
    parser.add_argument(
        "--data_path", default="features.npz", type=str,
        help="Path to your feature dataset (.npz file)"
    )
    # K argument for the FIXED K validation plot
    parser.add_argument(
        "--K", type=int, default=11, # Default reasonable for KNN/KMeans clusters
        help="Hyperparameter K for FIXED K validation plot (neighbors for KNN, clusters for KMeans)"
    )
    # Max K for the SWEEP K validation plot
    parser.add_argument(
        "--max_k_sweep", type=int, default=80,
        help="Maximum K value for the BEST K validation plot sweep (1 to max_k_sweep)"
    )
    parser.add_argument(
        "--cv_folds", type=int, default=5,
        help="Number of folds for KNN cross-validation plot (0 disables)"
    )
    # Add arguments for output filenames
    parser.add_argument(
        "--cv_plot_file", default="knn_cv_accuracy_plot.png", type=str,
        help="Filename for the KNN CV plot"
    )
    parser.add_argument(
        "--val_plot_file", default="validation_split_accuracy_plot.png", type=str,
        help="Filename for the FIXED K validation split plot"
    )
    parser.add_argument(
        "--best_k_val_plot_file", default="best_k_validation_split_plot.png", type=str,
        help="Filename for the BEST K validation split plot"
    )

    args = parser.parse_args()

    # --- Load Data ---
    print(f"Loading data from: {args.data_path}")
    try:
        feature_data = np.load(args.data_path, allow_pickle=True)
        xtrain_orig, ytrain_orig = feature_data["xtrain"], feature_data["ytrain"]
        print(f"Data loaded. xtrain shape: {xtrain_orig.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at '{args.data_path}'. Exiting.")
        exit()
    except KeyError:
        print(f"Error: Missing 'xtrain' or 'ytrain' in '{args.data_path}'. Exiting.")
        exit()


    # --- Generate KNN CV Plot ---
    if args.method == "knn" and args.cv_folds > 0:
        knn_param_grid = [1, 3, 5, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99]
        # Filter grid if needed for faster plotting
        # knn_param_grid = [k for k in knn_param_grid if k <= 21]
        plot_knn_cv(xtrain_orig, ytrain_orig, args.cv_folds, knn_param_grid, filename=args.cv_plot_file)
    elif args.cv_folds > 0:
        print("\n--- Skipping CV Plot: Only implemented for KNN ---")


    # --- Generate FIXED K Validation Split Plot ---
    validation_percentages = [0.15,  0.18, 0.20, 0.22, 0.25] # From main.py
    method_args_dict = {'K': args.K} # Pass the fixed K hyperparameter
    plot_validation_splits(xtrain_orig, ytrain_orig, args.method, method_args_dict,
                           validation_percentages, filename=args.val_plot_file)

    # --- Generate BEST K Validation Split Plot ---
    k_sweep_range = list(range(1, args.max_k_sweep + 1))
    plot_best_k_validation_splits(xtrain_orig, ytrain_orig, args.method, k_sweep_range,
                                  validation_percentages, filename=args.best_k_val_plot_file)


    print("\n--- Plot generation complete ---")