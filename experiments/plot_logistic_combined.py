import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

from src.methods.logistic_regression import LogisticRegression
from src.utils import accuracy_fn, normalize_fn, append_bias_term

np.random.seed(100) # Consistent seed

def stratified_kfold_indices(y, n_folds=5, seed=0):
    """
    Yields (train_idx, valid_idx) tuples for stratified k‑fold CV.
    Does not shuffle input arrays in‑place.
    """
    rng = np.random.default_rng(seed)
    labels = np.unique(y)
    per_class_idx = {lbl: np.where(y == lbl)[0] for lbl in labels}
    for lbl in labels:
        rng.shuffle(per_class_idx[lbl])

    folds = [list() for _ in range(n_folds)]
    for lbl, idxs in per_class_idx.items():
        for i, idx in enumerate(idxs):
            folds[i % n_folds].append(idx)

    for k in range(n_folds):
        valid_idx = np.array(folds[k])
        train_idx = np.array([idx
                              for fold_idx, fold in enumerate(folds) if fold_idx != k
                              for idx in fold])
        yield train_idx, valid_idx

def test_logistic_combined(xtrain_orig, ytrain_orig, xtest_orig, ytest_orig, lr_values, max_iters_values, n_folds, cv_seed=0):
    """
    Test logistic regression hyperparameters using both k-fold CV and test set evaluation.
    Calculates CV accuracy, test accuracy, and their mean for each parameter pair.
    """
    print(f"\n--- Evaluating Logistic Regression: Combining {n_folds}-Fold CV and Test Accuracy ---")
    num_lrs = len(lr_values)
    num_iters = len(max_iters_values)
    results = {
        'mean_cv_accuracy': np.zeros((num_lrs, num_iters)),
        'test_accuracy': np.zeros((num_lrs, num_iters)),
        'combined_mean_accuracy': np.zeros((num_lrs, num_iters))
    }

    total_combinations = num_lrs * num_iters
    current_combination = 0

    # Preprocessing stats based on the entire original training set for test evaluation
    means_train_full = np.mean(xtrain_orig, axis=0)
    stds_train_full = np.std(xtrain_orig, axis=0)
    stds_train_full[stds_train_full == 0] = 1e-8 # Avoid division by zero
    xtrain_full_proc = normalize_fn(xtrain_orig.copy(), means_train_full, stds_train_full)
    xtest_full_proc = normalize_fn(xtest_orig.copy(), means_train_full, stds_train_full)
    xtrain_full_proc = append_bias_term(xtrain_full_proc)
    xtest_full_proc = append_bias_term(xtest_full_proc)

    for i, lr in enumerate(lr_values):
        for j, max_iters in enumerate(max_iters_values):
            current_combination += 1
            print(f"[{current_combination}/{total_combinations}] Testing: lr={lr:.1e}, max_iters={max_iters}")

            # --- Cross-Validation ---
            fold_accuracies = []
            for fold_num, (train_idx, valid_idx) in enumerate(stratified_kfold_indices(ytrain_orig, n_folds, seed=cv_seed)):
                xtr_fold, xval_fold = xtrain_orig[train_idx], xtrain_orig[valid_idx]
                ytr_fold, yval_fold = ytrain_orig[train_idx], ytrain_orig[valid_idx]

                # Preprocessing within the fold
                means_fold = np.mean(xtr_fold, axis=0)
                stds_fold = np.std(xtr_fold, axis=0)
                stds_fold[stds_fold == 0] = 1e-8
                xtr_fold_proc = normalize_fn(xtr_fold, means_fold, stds_fold)
                xval_fold_proc = normalize_fn(xval_fold, means_fold, stds_fold)
                xtr_fold_proc = append_bias_term(xtr_fold_proc)
                xval_fold_proc = append_bias_term(xval_fold_proc)

                model_cv = LogisticRegression(lr=lr, max_iters=max_iters)
                model_cv.fit(xtr_fold_proc, ytr_fold)
                preds_val_fold = model_cv.predict(xval_fold_proc)
                fold_accuracies.append(accuracy_fn(preds_val_fold, yval_fold))

            mean_cv_acc = np.mean(fold_accuracies)
            results['mean_cv_accuracy'][i, j] = mean_cv_acc
            # print(f"  Avg CV Accuracy: {mean_cv_acc:.4f}") # Optional detail

            # --- Test Set Evaluation ---
            model_test = LogisticRegression(lr=lr, max_iters=max_iters)
            model_test.fit(xtrain_full_proc, ytrain_orig) # Train on fully preprocessed training data
            preds_test = model_test.predict(xtest_full_proc) # Predict on fully preprocessed test data
            test_acc = accuracy_fn(preds_test, ytest_orig)
            results['test_accuracy'][i, j] = test_acc
            # print(f"  Test Accuracy: {test_acc:.4f}") # Optional detail

            # --- Combined Metric ---
            combined_mean_acc = (mean_cv_acc + test_acc) / 2.0
            results['combined_mean_accuracy'][i, j] = combined_mean_acc
            print(f"  Combined Mean Accuracy: {combined_mean_acc:.4f}")

    return results

def plot_combined_results(lr_values, max_iters_values, results, n_folds):
    """
    Create 3D plot of Combined Mean Accuracy surface.
    """
    accuracy_key = 'combined_mean_accuracy'
    plot_title = f'Combined Mean Accuracy (Test & {n_folds}-Fold CV) Surface'
    save_filename = 'logistic_combined_mean_accuracy_surface.png'
    z_label = 'Combined Mean Accuracy'

    # Find best parameters based on combined mean accuracy
    best_idx = np.unravel_index(np.argmax(results[accuracy_key]), results[accuracy_key].shape)
    best_lr = lr_values[best_idx[0]]
    best_max_iters = max_iters_values[best_idx[1]]
    best_combined_accuracy = results[accuracy_key][best_idx]
    corresponding_cv_acc = results['mean_cv_accuracy'][best_idx]
    corresponding_test_acc = results['test_accuracy'][best_idx]

    # 3D Surface Plot
    fig = plt.figure(figsize=(14, 9)) # Slightly larger figure
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for surface plot
    X, Y = np.meshgrid(np.log10(lr_values), max_iters_values)
    Z = results[accuracy_key].T # Transpose to match X, Y shape

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.85)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label=z_label)

    # Plot best point
    best_x = np.log10(best_lr)
    best_y = best_max_iters
    best_z = best_combined_accuracy
    ax.scatter(best_x, best_y, best_z, color='red', s=120, label=f'Best Point ({best_combined_accuracy:.3f})', depthshade=False)

    # Add projection lines
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    z_lim = ax.get_zlim()
    z_min = z_lim[0]

    ax.plot([best_x, best_x], [best_y, best_y], [z_min, best_z], 'r--', alpha=0.6) # Vertical line
    ax.plot([x_lim[0], best_x], [best_y, best_y], [best_z, best_z], 'r--', alpha=0.6) # To YZ plane
    ax.plot([best_x, best_x], [y_lim[0], best_y], [best_z, best_z], 'r--', alpha=0.6) # To XZ plane

    # Add text labels for the best point coordinates
    ax.text(best_x, best_y, z_min, f'Iter={best_max_iters}', color='red', ha='center', va='top')
    ax.text(x_lim[0], best_y, best_z, f'Comb Acc={best_combined_accuracy:.3f}', color='red', ha='left', va='center')
    ax.text(best_x, y_lim[0], best_z, f'logLR={best_x:.1f}', color='red', ha='center', va='bottom')


    # Customize the view and labels
    ax.view_init(elev=25, azim=-135) # Adjusted view angle
    ax.set_xlabel('Log10(Learning Rate)')
    ax.set_ylabel('Max Iterations')
    ax.set_zlabel(z_label)
    ax.set_title(plot_title)
    plt.savefig(save_filename)
    print(f"\nPlot saved to {save_filename}")
    plt.close()

    # Print best parameters found
    print(f"\nBest parameters found based on Combined Mean Accuracy:")
    print(f"  Learning Rate: {best_lr:.1e}")
    print(f"  Max Iterations: {best_max_iters}")
    print(f"  Highest Combined Mean Accuracy: {best_combined_accuracy:.4f}")
    print(f"    (Achieved with Mean CV Accuracy: {corresponding_cv_acc:.4f} and Test Accuracy: {corresponding_test_acc:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Plot Combined Mean (CV & Test) Accuracy for Logistic Regression.")
    parser.add_argument('--data_path', default='Data_MS1_2025/features.npz', type=str, help='Path to features.npz file.')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of folds for cross-validation.')
    parser.add_argument('--cv_seed', type=int, default=42, help='Random seed for CV splits.')
    parser.add_argument('--lr_min_exp', type=int, default=-8, help='Min exponent for learning rate grid (10^x).')
    parser.add_argument('--lr_max_exp', type=int, default=-1, help='Max exponent for learning rate grid (10^x).')
    parser.add_argument('--iter_start', type=int, default=125, help='Start value for max_iters grid.')
    parser.add_argument('--iter_stop', type=int, default=3000, help='Stop value (exclusive) for max_iters grid.')
    parser.add_argument('--iter_step', type=int, default=125, help='Step value for max_iters grid.')
    args = parser.parse_args()

    if args.cv_folds <= 1:
        raise ValueError("Number of folds (cv_folds) must be greater than 1.")
    if not os.path.exists(args.data_path):
         raise FileNotFoundError(f"Data file not found at {args.data_path}")

    # Load data
    print(f"Loading data from {args.data_path}...")
    feature_data = np.load(args.data_path, allow_pickle=True)
    xtrain_orig, ytrain_orig = feature_data["xtrain"], feature_data["ytrain"]
    xtest_orig, ytest_orig = feature_data["xtest"], feature_data["ytest"]
    print(f"Loaded {xtrain_orig.shape[0]} training samples and {xtest_orig.shape[0]} test samples.")

    # Define parameter grids
    lr_values = np.array([10.0**i for i in range(args.lr_min_exp, args.lr_max_exp + 1)])
    max_iters_values = np.arange(args.iter_start, args.iter_stop, args.iter_step, dtype=int)
    if len(lr_values) == 0 or len(max_iters_values) == 0:
        raise ValueError("Parameter grids cannot be empty. Check grid arguments.")

    # Perform combined evaluation
    results = test_logistic_combined(
        xtrain_orig, ytrain_orig, xtest_orig, ytest_orig,
        lr_values, max_iters_values, args.cv_folds, args.cv_seed
    )

    # Plot the results
    plot_combined_results(lr_values, max_iters_values, results, args.cv_folds)

if __name__ == "__main__":
    main() 