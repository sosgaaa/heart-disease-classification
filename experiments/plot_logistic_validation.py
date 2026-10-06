import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from src.methods.logistic_regression import LogisticRegression
from src.utils import accuracy_fn, normalize_fn, append_bias_term
import sys  # For exiting if no valid split found

np.random.seed(100) # Use the same seed as main.py for consistency

def stratified_kfold_indices(y, n_folds=5, seed=0):
    """
    Yields (train_idx, valid_idx) tuples for stratified k‑fold CV.
    Does not shuffle input arrays in‑place.
    Adapted from main.py.
    """
    rng = np.random.default_rng(seed)
    # group indices by class label
    labels = np.unique(y)
    per_class_idx = {lbl: np.where(y == lbl)[0] for lbl in labels}
    for lbl in labels:
        rng.shuffle(per_class_idx[lbl]) # shuffle in each class bucket

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

def test_logistic_cv_params(xtrain_for_cv, ytrain_for_cv, lr_values, max_iters_values, n_folds, cv_seed=42):
    """
    Test logistic regression hyperparameters using k-fold cross-validation
    on the provided training data split.
    Returns a dictionary containing the mean CV accuracy for each parameter pair.
    """
    print(f"\n--- Performing {n_folds}-Fold Cross-Validation on Selected Train Split ---")
    cv_results = {
        'mean_cv_accuracy': np.zeros((len(lr_values), len(max_iters_values)))
    }

    total_combinations = len(lr_values) * len(max_iters_values)
    current_combination = 0

    for i, lr in enumerate(lr_values):
        for j, max_iters in enumerate(max_iters_values):
            current_combination += 1
            print(f"[{current_combination}/{total_combinations}] Testing CV: lr={lr:.1e}, max_iters={max_iters}")

            fold_accuracies = []
            # Perform k-fold cross-validation using the *selected* training data
            for fold_num, (train_idx, valid_idx) in enumerate(stratified_kfold_indices(ytrain_for_cv, n_folds, seed=cv_seed)):
                # These indices refer to the *subset* `xtrain_for_cv` / `ytrain_for_cv`
                xtr, xval = xtrain_for_cv[train_idx], xtrain_for_cv[valid_idx]
                ytr, yval = ytrain_for_cv[train_idx], ytrain_for_cv[valid_idx]

                # Preprocessing within the fold (using only fold's training data stats)
                means = np.mean(xtr, axis=0)
                stds = np.std(xtr, axis=0)
                stds[stds == 0] = 1e-8 # Avoid division by zero

                xtr_proc = normalize_fn(xtr, means, stds)
                xval_proc = normalize_fn(xval, means, stds)

                xtr_proc = append_bias_term(xtr_proc)
                xval_proc = append_bias_term(xval_proc)

                # Initialize and train model
                model = LogisticRegression(lr=lr, max_iters=max_iters)
                model.fit(xtr_proc, ytr)
                preds_val = model.predict(xval_proc)

                # Calculate accuracy for this fold
                fold_acc = accuracy_fn(preds_val, yval)
                fold_accuracies.append(fold_acc)
                # Optional: print fold accuracy
                # print(f"  Fold {fold_num+1}/{n_folds} accuracy: {fold_acc:.3f}%")

            # Calculate mean accuracy across folds for this parameter combination
            mean_acc = np.mean(fold_accuracies)
            cv_results['mean_cv_accuracy'][i, j] = mean_acc
            print(f"  Avg CV Accuracy: {mean_acc:.4f}")

    return cv_results

def plot_cv_results(lr_values, max_iters_values, cv_results, n_folds, best_split_perc):
    """
    Create 3D plot of Mean Cross-Validated Accuracy surface based on the best split.
    (Adapted from previous plotting function)
    """
    accuracy_key = 'mean_cv_accuracy' # Use mean CV accuracy
    plot_title = f'Mean CV Accuracy Surface ({n_folds}-Fold CV on {best_split_perc*100:.0f}% Train Split)'
    save_filename = f'logistic_cv_accuracy_surface_split_{best_split_perc*100:.0f}.png'
    z_label = 'Mean CV Accuracy'

    # Find best parameters based on mean CV accuracy
    # Find the maximum accuracy achieved
    max_accuracy = np.max(cv_results[accuracy_key])

    # Find all indices (i for lr, j for max_iters) where accuracy is maximal
    max_indices = np.argwhere(cv_results[accuracy_key] == max_accuracy)

    # Select the best index based on: min max_iters, then min lr
    best_combo_idx = max_indices[0] # Start with the first one found
    min_iters = max_iters_values[best_combo_idx[1]]
    min_lr = lr_values[best_combo_idx[0]]

    for idx in max_indices:
        current_iters = max_iters_values[idx[1]]
        current_lr = lr_values[idx[0]]

        if current_iters < min_iters:
            min_iters = current_iters
            min_lr = current_lr
            best_combo_idx = idx
        elif current_iters == min_iters:
            if current_lr < min_lr:
                min_lr = current_lr
                best_combo_idx = idx

    best_idx = tuple(best_combo_idx)

    # --- Original code using simple argmax (commented out) ---
    # best_idx = np.unravel_index(np.argmax(cv_results[accuracy_key]), cv_results[accuracy_key].shape)
    # ---

    best_lr = lr_values[best_idx[0]]
    best_max_iters = max_iters_values[best_idx[1]]
    best_mean_accuracy = cv_results[accuracy_key][best_idx]

    # 3D Surface Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for surface plot
    X, Y = np.meshgrid(np.log10(lr_values), max_iters_values)
    Z = cv_results[accuracy_key].T # Transpose to match X, Y shape

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Plot best point
    best_x = np.log10(best_lr)
    best_y = best_max_iters
    best_z = best_mean_accuracy
    ax.scatter(best_x, best_y, best_z, color='red', s=100, label=f'Best Point ({best_mean_accuracy:.3f})')

    # Add projection lines
    # Get axis limits *after* plotting the surface
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    z_lim = ax.get_zlim()
    z_min = z_lim[0]

    ax.plot([best_x, best_x], [best_y, best_y], [z_min, best_z], 'r--', alpha=0.5) # Vertical line
    ax.plot([x_lim[0], best_x], [best_y, best_y], [best_z, best_z], 'r--', alpha=0.5) # To YZ plane
    ax.plot([best_x, best_x], [y_lim[0], best_y], [best_z, best_z], 'r--', alpha=0.5) # To XZ plane

    # Add text labels for the best point coordinates
    ax.text(best_x, best_y, z_min, f'Iter={best_max_iters}', color='red', ha='center', va='top')
    ax.text(x_lim[0], best_y, best_z, f'Mean Acc={best_mean_accuracy:.3f}', color='red', ha='left', va='center')
    ax.text(best_x, y_lim[0], best_z, f'logLR={best_x:.1f}', color='red', ha='center', va='bottom')

    # Customize the view and labels
    ax.view_init(elev=20, azim=45)
    ax.set_xlabel('Log10(Learning Rate)')
    ax.set_ylabel('Max Iterations')
    ax.set_zlabel(z_label)
    ax.set_title(plot_title)
    plt.savefig(save_filename)
    print(f"\nPlot saved to {save_filename}")
    plt.close()

    # Print best parameters found
    print(f"\nBest parameters found based on {n_folds}-Fold Cross-Validation (on {best_split_perc*100:.0f}% Train Split):")
    print(f"Learning Rate: {best_lr:.1e}")
    print(f"Max Iterations: {best_max_iters}")
    print(f"Mean CV Accuracy: {best_mean_accuracy:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Find best validation split, then plot Logistic Regression CV results.")
    parser.add_argument('--data_path', default='Data_MS1_2025/features.npz', type=str)
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of folds for hyperparameter cross-validation.')
    parser.add_argument('--cv_seed', type=int, default=42, help='Random seed for CV splits.')
    parser.add_argument('--split_seed', type=int, default=43, help='Random seed for initial train/validation split.') # Different seed for split

    # Validation split parameters
    parser.add_argument('--validation_percentages', nargs='+', type=float, default=[0.15, 0.20, 0.25], help='List of validation percentages to test for the initial split.')
    parser.add_argument('--split_test_lr_exp', type=int, default=-5, help='Log10 of fixed LR for testing splits (10^x).')
    parser.add_argument('--split_test_max_iters', type=int, default=500, help='Fixed max_iters for testing splits.')

    # Hyperparameter grid search parameters
    parser.add_argument('--lr_min_exp', type=int, default=-8, help='Min exponent for learning rate grid (10^x).')
    parser.add_argument('--lr_max_exp', type=int, default=-1, help='Max exponent for learning rate grid (10^x).')
    parser.add_argument('--iter_start', type=int, default=125, help='Start value for max_iters grid.')
    parser.add_argument('--iter_stop', type=int, default=3000, help='Stop value (exclusive) for max_iters grid.')
    parser.add_argument('--iter_step', type=int, default=125, help='Step value for max_iters grid.')

    args = parser.parse_args()

    if args.cv_folds <= 1:
        raise ValueError("Number of folds (cv_folds) must be greater than 1.")
    if not args.validation_percentages:
        raise ValueError("Must provide at least one validation percentage to test.")

    # Load data
    print(f"Loading data from {args.data_path}...")
    feature_data = np.load(args.data_path, allow_pickle=True)
    xtrain_orig, ytrain_orig = feature_data["xtrain"], feature_data["ytrain"]
    print(f"Loaded {xtrain_orig.shape[0]} original training samples.")

    # --- Step 1: Find the best validation split percentage ---
    print("\n--- Finding Best Initial Train/Validation Split ---")
    results_per_split = {}
    split_rng = np.random.default_rng(args.split_seed) # Use separate RNG for splitting
    fixed_lr_for_split_test = 10.0**args.split_test_lr_exp

    for p in args.validation_percentages:
        validation_perc = p
        train_perc = 1.0 - validation_perc
        print(f"\n--- Testing Split: {train_perc*100:.0f}% Train / {validation_perc*100:.0f}% Validation ---")

        # Simple random split (stratification might be better but keep consistent with main.py example)
        num_train_samples = xtrain_orig.shape[0]
        validation_size = int(num_train_samples * validation_perc)
        if validation_size == 0 or num_train_samples - validation_size == 0:
            print(f"Warning: Training data size ({num_train_samples}) too small for {validation_perc*100:.0f}% split. Skipping.")
            continue

        shuffled_indices = split_rng.permutation(num_train_samples)
        valid_indices = shuffled_indices[:validation_size]
        train_indices = shuffled_indices[validation_size:]

        xtrain_split = xtrain_orig[train_indices]
        ytrain_split = ytrain_orig[train_indices]
        xvalid_split = xtrain_orig[valid_indices]
        yvalid_split = ytrain_orig[valid_indices]

        print(f"Split sizes: Train={xtrain_split.shape[0]}, Validation={xvalid_split.shape[0]}")

        # Preprocessing based on this *split's* training data
        means_split = np.mean(xtrain_split, axis=0)
        stds_split = np.std(xtrain_split, axis=0)
        stds_split[stds_split == 0] = 1e-8 # Avoid division by zero

        xtrain_split_proc = normalize_fn(xtrain_split, means_split, stds_split)
        xvalid_split_proc = normalize_fn(xvalid_split, means_split, stds_split)

        xtrain_split_proc = append_bias_term(xtrain_split_proc)
        xvalid_split_proc = append_bias_term(xvalid_split_proc)

        # Train a model with fixed hyperparameters
        split_test_model = LogisticRegression(lr=fixed_lr_for_split_test, max_iters=args.split_test_max_iters)
        print(f"Fitting logistic model (lr={fixed_lr_for_split_test:.1e}, iters={args.split_test_max_iters})...")
        split_test_model.fit(xtrain_split_proc, ytrain_split)
        preds_valid_split = split_test_model.predict(xvalid_split_proc)

        # Evaluate accuracy on the validation part of this split
        acc_valid = accuracy_fn(preds_valid_split, yvalid_split)
        print(f"  Validation Accuracy for this split: {acc_valid:.4f}")

        results_per_split[p] = {
            'accuracy': acc_valid,
            'train_indices': train_indices,
            'valid_indices': valid_indices
        }

    if not results_per_split:
        print("Error: No valid splits were tested. Exiting.")
        sys.exit(1)

    # Find the best split percentage
    best_split_perc = max(results_per_split, key=lambda p: results_per_split[p]['accuracy'])
    best_accuracy = results_per_split[best_split_perc]['accuracy']
    print(f"\n--- Best validation split percentage: {best_split_perc*100:.0f}% (Yielded Accuracy: {best_accuracy:.4f}) ---")

    # Get the training data corresponding to the best split
    best_train_indices = results_per_split[best_split_perc]['train_indices']
    xtrain_for_cv = xtrain_orig[best_train_indices]
    ytrain_for_cv = ytrain_orig[best_train_indices]
    print(f"Using {xtrain_for_cv.shape[0]} samples for subsequent {args.cv_folds}-Fold CV hyperparameter search.")


    # --- Step 2: Perform cross-validation hyperparameter search on the selected training data ---
    # Define hyperparameter grids from args
    lr_values = np.array([10.0**i for i in range(args.lr_min_exp, args.lr_max_exp + 1)])
    max_iters_values = np.arange(args.iter_start, args.iter_stop, args.iter_step, dtype=int)

    # Perform cross-validation parameter search using the *best training split*
    cv_results = test_logistic_cv_params(
        xtrain_for_cv, ytrain_for_cv, lr_values, max_iters_values, args.cv_folds, args.cv_seed
    )

    # --- Step 3: Plot the results ---
    plot_cv_results(lr_values, max_iters_values, cv_results, args.cv_folds, best_split_perc=(1.0-best_split_perc))


if __name__ == "__main__":
    main() 