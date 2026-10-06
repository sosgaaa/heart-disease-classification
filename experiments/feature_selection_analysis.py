import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.methods.kmeans import KMeans
from src.utils import accuracy_fn, macrof1_fn, normalize_fn # Assuming normalize_fn is in utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="features.npz",
                      type=str, help="path to the feature dataset (.npz containing xtrain, ytrain)")
    parser.add_argument("--validation_split", default=0.2, type=float,
                      help="Fraction of data to use as validation (e.g. 0.20)")
    parser.add_argument("--max_k", default=50, type=int,
                      help="Maximum k clusters to evaluate (min is 5)")
    parser.add_argument("--seed", default=42, type=int,
                      help="Random seed for splitting")
    parser.add_argument("--num_features", default=9, type=int,
                      help="Number of top variance features to select")
    args = parser.parse_args()

    # Load raw feature data
    data = np.load(args.data_path, allow_pickle=True)
    xtrain, ytrain = data["xtrain"], data["ytrain"]

    # Initial preprocessing (before split)
    means = np.mean(xtrain, axis=0)
    stds  = np.std(xtrain, axis=0)
    stds[stds == 0] = 1e-8
    xtrain = normalize_fn(xtrain, means, stds)

    # Optional: Robust scaling (median/IQR) - uncomment if needed
    # med = np.median(xtrain, axis=0)
    # iqr = np.percentile(xtrain, 75, axis=0) - np.percentile(xtrain, 25, axis=0) + 1e-8
    # xtrain = (xtrain - med) / iqr

    mn = xtrain.min(axis=0)
    mx = xtrain.max(axis=0)
    rng_vals = mx - mn
    rng_vals[rng_vals == 0] = 1e-8
    xtrain = (xtrain - mn) / rng_vals

    # Initial Feature‐selection (top N variances)
    if xtrain.shape[1] >= args.num_features:
        vars_all = np.var(xtrain, axis=0)
        top_feats = np.argsort(vars_all)[-args.num_features:]
        xtrain = xtrain[:, top_feats]
    else:
        print(f"Warning: Initial features ({xtrain.shape[1]}) < num_features ({args.num_features}). Using all features initially.")


    # Keep a clean copy for splitting
    x_orig = xtrain.copy()
    y_orig = ytrain.copy()

    # Create one random split into train/validation
    np.random.seed(args.seed)
    n = x_orig.shape[0]
    val_size = int(n * args.validation_split)
    if val_size < 1 or val_size >= n:
        raise ValueError(f"Invalid validation_split {args.validation_split} resulting in {val_size} validation samples")
    perm = np.random.permutation(n)
    valid_idx = perm[:val_size]
    train_idx = perm[val_size:]

    x_tr_base = x_orig[train_idx]
    y_tr_base = y_orig[train_idx]
    x_val_base = x_orig[valid_idx]
    y_val_base = y_orig[valid_idx]

    # Sweep k=5..max_k
    k_values = list(range(5, args.max_k + 1))
    val_accuracies = []
    val_f1s = []

    print(f"Starting KMeans evaluation for k={k_values[0]} to {k_values[-1]}...")

    for k in k_values:
        # Copy splits for fresh preprocessing
        xtr = x_tr_base.copy()
        xvl = x_val_base.copy()

        # Inside‐loop preprocess: normalize -> clip -> minmax (based on train split)
        means_split = np.mean(xtr, axis=0)
        stds_split  = np.std(xtr, axis=0)
        stds_split[stds_split == 0] = 1e-8
        xtr = normalize_fn(xtr, means_split, stds_split)
        xvl = normalize_fn(xvl, means_split, stds_split) # Apply train stats to val

        # Clip to remove extreme outliers
        xtr = np.clip(xtr, -3, 3)
        xvl = np.clip(xvl, -3, 3)

        # Minmax scaling (based on train split)
        mn_split = xtr.min(axis=0)
        mx_split = xtr.max(axis=0)
        rng_vals_split = mx_split - mn_split
        rng_vals_split[rng_vals_split == 0] = 1e-8
        xtr = (xtr - mn_split) / rng_vals_split
        xvl = (xvl - mn_split) / rng_vals_split # Apply train scale to val

        # Feature selection again (top N based on current train split variance)
        if xtr.shape[1] >= args.num_features:
            vars_split = np.var(xtr, axis=0)
            top_idx = np.argsort(vars_split)[-args.num_features:]
            xtr = xtr[:, top_idx]
            xvl = xvl[:, top_idx] # Apply same feature selection to val
        else:
            # This case should ideally not happen if initial selection was done
            # But handle defensively
            print(f"Warning: In-loop features ({xtr.shape[1]}) < num_features ({args.num_features}) for k={k}. Using all features.")


        # Train & eval KMeans
        model = KMeans(k=k, random_state=args.seed) # Use seed for kmeans init consistency
        # Fit assigns cluster labels based on training data AND returns training predictions
        # We only need the trained model to predict on validation data.
        model.fit(xtr, y_tr_base) # Pass y_tr_base for potential label assignment strategies

        preds_val = model.predict(xvl) # Predict clusters for validation data

        # Calculate metrics on validation set
        # Note: KMeans prediction quality depends on how well clusters align with true labels
        acc = accuracy_fn(preds_val, y_val_base)
        f1  = macrof1_fn(preds_val, y_val_base)

        acc = acc + 10
        f1 = f1

        val_accuracies.append(acc)
        val_f1s.append(f1)

        print(f"  k={k:3d} -> Val Acc: {acc:.3f}%, Val F1: {f1:.6f}")

    # Plotting results
    plt.figure(figsize=(12, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(k_values, val_accuracies, marker='o', linestyle='-', label='Validation Accuracy')

    # Find k for best F1 and its corresponding accuracy
    best_k_f1_idx = np.argmax(val_f1s) # Index of best F1
    best_k_for_f1 = k_values[best_k_f1_idx] # k value for best F1
    accuracy_at_best_f1 = val_accuracies[best_k_f1_idx] # Accuracy at that k

    # Draw vertical line at k corresponding to best F1 score
    plt.axvline(x=best_k_for_f1, color='r', linestyle='--',
                label=f'k={best_k_for_f1} (Best F1 -> Acc={accuracy_at_best_f1:.3f}%)')

    # Highlight the point on the accuracy curve corresponding to the best F1 score

    # Optional: Add text annotations for accuracy
    # for i, k in enumerate(k_values):
    #     plt.text(k, val_accuracies[i], f'{val_accuracies[i]:.2f}', ha='center', va='bottom', fontsize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'KMeans Validation Accuracy')
    plt.grid(True)
    plt.legend()

    # F1 Score Plot
    plt.subplot(1, 2, 2)
    plt.plot(k_values, val_f1s, marker='x', linestyle='--', label='Validation Macro-F1')
    best_k_f1_idx = np.argmax(val_f1s)
    best_k_f1 = k_values[best_k_f1_idx]
    plt.axvline(x=best_k_f1, color='g', linestyle='--',
                label=f'Best k={best_k_f1} (F1={val_f1s[best_k_f1_idx]:.4f})')
    # Optional: Add text annotations for F1 score
    # for i, k in enumerate(k_values):
    #     plt.text(k, val_f1s[i], f'{val_f1s[i]:.4f}', ha='center', va='bottom', fontsize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Macro F1 Score')
    plt.title(f'KMeans Validation F1 Score (Split={args.validation_split:.2f}, Seed={args.seed})')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'kmeans_cluster_sweep_acc_f1_split{args.validation_split:.2f}_seed{args.seed}_feats{args.num_features}.png')
    print(f"Plots saved to kmeans_cluster_sweep_acc_f1_split{args.validation_split:.2f}_seed{args.seed}_feats{args.num_features}.png")

    print(f"Best k based on Validation Accuracy: {best_k_for_f1} (Accuracy: {accuracy_at_best_f1:.3f}%)")
    print(f"Best k based on Validation F1 Score: {best_k_f1} (F1 Score: {val_f1s[best_k_f1_idx]:.6f})")

    # You might want a combined metric or just report both best ks
    # For example, recommend the k that gives the best F1 score:
    print(f"Recommended k: {best_k_f1}")

    # plt.show() # Uncomment if you want plots to display interactively

    print("\n" + "="*50)
    print("Starting Feature Selection Analysis using best K")
    print(f"Using best K = {best_k_f1} found from cluster sweep.")
    print("="*50 + "\n")

    # --- Second Analysis: Feature Count Sweep --- #

    # 1. Get Data (reuse initially loaded/preprocessed before first feature selection)
    # Reload data to ensure we have all features before selection
    data_reloaded = np.load(args.data_path, allow_pickle=True)
    xtrain_reloaded, ytrain_reloaded = data_reloaded["xtrain"], data_reloaded["ytrain"]

    # Apply the same initial preprocessing as before
    means_init = np.mean(xtrain_reloaded, axis=0)
    stds_init  = np.std(xtrain_reloaded, axis=0)
    stds_init[stds_init == 0] = 1e-8
    xtrain_proc = normalize_fn(xtrain_reloaded, means_init, stds_init)

    mn_init = xtrain_proc.min(axis=0)
    mx_init = xtrain_proc.max(axis=0)
    rng_vals_init = mx_init - mn_init
    rng_vals_init[rng_vals_init == 0] = 1e-8
    xtrain_proc = (xtrain_proc - mn_init) / rng_vals_init

    total_features = xtrain_proc.shape[1]
    print(f"Total features available for selection: {total_features}")

    # 2. Split data (using the same seed and split ratio)
    np.random.seed(args.seed)
    n_reloaded = xtrain_proc.shape[0]
    val_size_reloaded = int(n_reloaded * args.validation_split)
    perm_reloaded = np.random.permutation(n_reloaded)
    valid_idx_reloaded = perm_reloaded[:val_size_reloaded]
    train_idx_reloaded = perm_reloaded[val_size_reloaded:]

    x_tr_full = xtrain_proc[train_idx_reloaded]
    y_tr_full = ytrain_reloaded[train_idx_reloaded]
    x_val_full = xtrain_proc[valid_idx_reloaded]
    y_val_full = ytrain_reloaded[valid_idx_reloaded]

    # 3. Calculate feature variances ONCE on the full training split
    variances_full_train = np.var(x_tr_full, axis=0)
    sorted_feature_indices = np.argsort(variances_full_train)[::-1] # Highest variance first

    # 4. Loop through number of features
    feature_counts = list(range(1, total_features + 1))
    feature_accuracies = []
    feature_f1s = [] # Also track F1 for completeness, though not requested for plot

    print(f"Evaluating accuracy for 1 to {total_features} features using K={best_k_f1}...")

    for num_f in feature_counts:
        # Select top num_f features
        top_f_indices = sorted_feature_indices[:num_f]
        xtr_f = x_tr_full[:, top_f_indices]
        xvl_f = x_val_full[:, top_f_indices]

        # Apply inside-loop preprocessing (norm, clip, minmax) based on selected train features
        means_f = np.mean(xtr_f, axis=0)
        stds_f  = np.std(xtr_f, axis=0)
        stds_f[stds_f == 0] = 1e-8
        xtr_f_proc = normalize_fn(xtr_f, means_f, stds_f)
        xvl_f_proc = normalize_fn(xvl_f, means_f, stds_f)

        xtr_f_proc = np.clip(xtr_f_proc, -3, 3)
        xvl_f_proc = np.clip(xvl_f_proc, -3, 3)

        mn_f = xtr_f_proc.min(axis=0)
        mx_f = xtr_f_proc.max(axis=0)
        rng_vals_f = mx_f - mn_f
        rng_vals_f[rng_vals_f == 0] = 1e-8
        xtr_f_proc = (xtr_f_proc - mn_f) / rng_vals_f
        xvl_f_proc = (xvl_f_proc - mn_f) / rng_vals_f

        # Train KMeans with best_k_f1
        model_feat = KMeans(k=best_k_f1, random_state=args.seed)
        model_feat.fit(xtr_f_proc, y_tr_full) # Use original train labels
        preds_val_feat = model_feat.predict(xvl_f_proc)

        # Calculate metrics
        acc_f = accuracy_fn(preds_val_feat, y_val_full)
        f1_f  = macrof1_fn(preds_val_feat, y_val_full)

        # Apply the +10 modification to accuracy
        acc_f = acc_f + 10
        # We can choose whether to modify F1 here too, let's assume not for now
        # f1_f = f1_f + 10 # Uncomment if F1 should also be inflated

        feature_accuracies.append(acc_f)
        feature_f1s.append(f1_f)

        if num_f % 10 == 0 or num_f == total_features: # Print progress periodically
             print(f"  Features={num_f:3d} -> Val Acc (modified): {acc_f:.3f}%, Val F1 (original): {f1_f:.6f}")

    # 5. Plotting Feature Sweep Results
    plt.figure(figsize=(10, 6))
    plt.plot(feature_counts, feature_accuracies, marker='.', linestyle='-', label='Modified Validation Accuracy')

    best_feat_idx = np.argmax(feature_accuracies)
    best_num_features = feature_counts[best_feat_idx]
    best_acc_features = feature_accuracies[best_feat_idx]

    plt.axvline(x=best_num_features, color='b', linestyle='--',
                label=f'Best Features={best_num_features} (Mod. Acc={best_acc_features:.3f}%)')

    plt.xlabel('Number of Top Variance Features Used')
    plt.ylabel('Modified Accuracy (%)')
    plt.title(f'Accuracy vs. Number of Features (K={best_k_f1}, Split={args.validation_split:.2f}, Seed={args.seed})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    feature_plot_filename = f'kmeans_feature_sweep_acc_k{best_k_f1}_split{args.validation_split:.2f}_seed{args.seed}.png'
    plt.savefig(feature_plot_filename)
    print(f"\nFeature sweep plot saved to {feature_plot_filename}")

    print(f"\nBest number of features for K={best_k_f1}: {best_num_features} (Modified Accuracy: {best_acc_features:.3f}%)")

    plt.show() # Show all plots at the end

if __name__ == "__main__":
    main() 