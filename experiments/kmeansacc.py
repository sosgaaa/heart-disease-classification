import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.methods.kmeans import KMeans
from src.utils import normalize_fn, accuracy_fn, macrof1_fn

def main(args):
    # 1) Load raw feature data
    data = np.load(args.data_path, allow_pickle=True)
    xtrain, ytrain = data["xtrain"], data["ytrain"]

    # DEBUG: Check raw loaded data
    print(f"\n[kmeansacc.py DEBUG Raw Load]")
    print(f"  xtrain shape: {xtrain.shape}, sum: {np.sum(xtrain):.4f}\n")

    # 2) One‐time preprocessing before any splits
    #    (matches your pipeline)
    means = np.mean(xtrain, axis=0)
    stds  = np.std(xtrain, axis=0)
    xtrain = normalize_fn(xtrain, means, stds)

    med = np.median(xtrain, axis=0)
    iqr = np.percentile(xtrain, 75, axis=0) - np.percentile(xtrain, 25, axis=0) + 1e-8
    xtrain = (xtrain - med) / iqr

    mn = xtrain.min(axis=0)
    mx = xtrain.max(axis=0)
    xtrain = (xtrain - mn) / (mx - mn + 1e-8)

    # Feature‐select top 7 variances (for KMeans)
    vars_all = np.var(xtrain, axis=0)
    top_feats = np.argsort(vars_all)[-7:]
    xtrain = xtrain[:, top_feats]

    # Keep a clean copy for splitting
    x_orig = xtrain.copy()
    y_orig = ytrain.copy()

    # 3) Do one random split into train/validation using legacy random for consistency with main.py's validation split logic
    np.random.seed(args.seed) # Use the legacy seed setting
    n = x_orig.shape[0]
    val_size = int(n * args.validation_split)
    if val_size < 1 or val_size >= n:
        raise ValueError(f"Invalid validation_split {args.validation_split}")
    perm = np.random.permutation(n) # Use legacy permutation
    valid_idx = perm[:val_size]
    train_idx = perm[val_size:]

    x_tr_base = x_orig[train_idx]
    y_tr_base = y_orig[train_idx]
    x_val_base = x_orig[valid_idx]
    y_val_base = y_orig[valid_idx]

    # 4) Sweep k=5..100
    k_values = list(range(5, args.max_k+1))
    val_accuracies = []
    val_f1s = []

    for k in k_values:
        # 4a) Copy splits for fresh preprocessing
        xtr = x_tr_base.copy()
        xvl = x_val_base.copy()

        # 4b) Inside‐loop preprocess: normalize → clip → minmax
        means = np.mean(xtr, axis=0)
        stds  = np.std(xtr, axis=0)
        stds[stds == 0] = 1e-8
        xtr = normalize_fn(xtr, means, stds)
        xvl = normalize_fn(xvl, means, stds)

        # clip to remove extreme outliers
        xtr = np.clip(xtr, -3, 3)
        xvl = np.clip(xvl, -3, 3)

        mn = xtr.min(axis=0)
        mx = xtr.max(axis=0)
        rng_vals = mx - mn
        rng_vals[rng_vals == 0] = 1e-8
        xtr = (xtr - mn) / rng_vals
        xvl = (xvl - mn) / rng_vals

        # 4c) Feature selection again (top 7)
        vars_split = np.var(xtr, axis=0)
        if xtr.shape[1] >= 7:
            top_idx = np.argsort(vars_split)[-7:]
        else:
            top_idx = slice(None)
        xtr = xtr[:, top_idx]
        xvl = xvl[:, top_idx]
        
        # DEBUG: Check data just before fit for k=34
        if k == 34:
            print(f"\n[kmeansacc.py DEBUG k={k}]")
            print(f"  xtr shape: {xtr.shape}, sum: {np.sum(xtr):.4f}")
            print(f"  y_tr_base shape: {y_tr_base.shape}")
            print(f"  xvl shape: {xvl.shape}, sum: {np.sum(xvl):.4f}")
            print(f"  y_val_base shape: {y_val_base.shape}\n")

        # 4d) Train & eval
        model = KMeans(k=k, random_state=42)
        preds_train = model.fit(xtr, y_tr_base)
        preds_val   = model.predict(xvl)

        acc = accuracy_fn(preds_val, y_val_base)
        f1  = macrof1_fn(preds_val, y_val_base)

        val_accuracies.append(acc)
        val_f1s.append(f1)

        print(f"k={k:3d} → val accuracy={acc:.3f}%, val F1={f1:.6f}")

    # 5) Plot
    plt.figure(figsize=(10,6))
    plt.plot(k_values, val_accuracies, marker='o', linestyle='-', label='Validation Accuracy')
    # Add text annotations for accuracy
    for i, k in enumerate(k_values):
        plt.text(k, val_accuracies[i], f'{val_accuracies[i]:.3f}', ha='center', va='bottom', fontsize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Accuracy (%)')
    plt.title(f'KMeans Validation Accuracy (split={int(args.validation_split*100)}%)')
    plt.grid(True)
    plt.legend()
    plt.savefig('kmeans_val_accuracy.png')

    plt.figure(figsize=(10,6))
    plt.plot(k_values, val_f1s, marker='x', linestyle='--', label='Validation Macro‑F1')
    # Add text annotations for F1 score
    for i, k in enumerate(k_values):
        plt.text(k, val_f1s[i], f'{val_f1s[i]:.4f}', ha='center', va='bottom', fontsize=8) # Adjusted format for F1
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Macro F1 Score')
    plt.title(f'KMeans Validation F1 (split={int(args.validation_split*100)}%)')
    plt.grid(True)
    plt.legend()
    plt.savefig('kmeans_val_f1.png')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default='features.npz',
        help='.npz file containing xtrain,ytrain'
    )
    parser.add_argument(
        '--validation_split',
        type=float,
        default=0.22,
        help='Fraction of data to use as validation (e.g. 0.20)'
    )
    parser.add_argument(
        '--max_k',
        type=int,
        default=100,
        help='Maximum k to evaluate (min is 5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=100,
        help='Random seed for splitting'
    )
    args = parser.parse_args()
    main(args)
