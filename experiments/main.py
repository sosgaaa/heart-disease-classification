import argparse

import numpy as np

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.knn import KNN
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os

np.random.seed(100)

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
    Returns best_param, cv_results_dict
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


def main(args):
    """
    Run an exploratory classification experiment with the requested configuration.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data

    # EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load(args.data_path, allow_pickle=True)
        xtrain, xtest = feature_data["xtrain"], feature_data["xtest"]
        ytrain, ytest = feature_data["ytrain"], feature_data["ytest"]

    # ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path, "dog-small-64")
        xtrain, xtest, ytrain, ytest = load_data(data_dir)

    ## 2. Then we must prepare it. This is where you can create a validation set, normalize, add bias, etc.
    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        print("\n--- Running in Validation Mode: Testing different validation splits ---")
        validation_percentages = [0.15, 0.18, 0.20, 0.22, 0.25] # Percentages to test
        # Fit standardization statistics on training data.
        means = np.mean(xtrain, axis=0)
        stds = np.std(xtrain, axis=0)

        # apply normalization
        xtrain = normalize_fn(xtrain, means, stds)
        xtest = normalize_fn(xtest, means, stds)

        # clip values to remove extreme outliers
        med  = np.median(xtrain, axis=0)
        iqr  = np.percentile(xtrain, 75, axis=0) - np.percentile(xtrain, 25, axis=0) + 1e-8
        xtrain = (xtrain - med) / iqr
        xtest  = (xtest  - med) / iqr


        # apply minmax scaling
        min_vals = xtrain.min(axis=0)
        max_vals = xtrain.max(axis=0)
        xtrain = (xtrain - min_vals) / (max_vals - min_vals + 1e-8)
        xtest = (xtest - min_vals) / (max_vals - min_vals + 1e-8)


        # handle features specifically for KMeans
        if args.method == "kmeans":
            variances = np.var(xtrain, axis=0)
            top_k_features = np.argsort(variances)[-7:]
            xtrain = xtrain[:, top_k_features]
            xtest = xtest[:, top_k_features]
        else:
            # for knn, just use top variance features
            variances = np.var(xtrain, axis=0)
            top_k_features = np.argsort(variances)[-10:]
            xtrain = xtrain[:, top_k_features]
            xtest = xtest[:, top_k_features]

        # keep original data safe
        xtrain_orig = xtrain.copy()
        ytrain_orig = ytrain.copy()
        #only validation split is used
        results_per_split = {}

        #find the best validation split by testing all percentages
        for p in validation_percentages:
            print(f"\n--- Testing Validation Split: {p*100:.0f}% ---")

            # split original training data for this percentage
            num_train_samples = xtrain_orig.shape[0]
            validation_size = int(num_train_samples * p)
            if validation_size == 0 or num_train_samples - validation_size == 0:
                print(f"Warning: Training data size ({num_train_samples}) too small for {p*100:.0f}% split. Skipping.")
                continue

            shuffled_indices = np.random.permutation(num_train_samples)
            valid_indices = shuffled_indices[:validation_size]
            train_indices = shuffled_indices[validation_size:]

            xtrain_split = xtrain_orig[train_indices]
            ytrain_split = ytrain_orig[train_indices]
            xvalid_split = xtrain_orig[valid_indices]
            yvalid_split = ytrain_orig[valid_indices]
            print(f"Split sizes: Train={xtrain_split.shape[0]}, Validation={xvalid_split.shape[0]}")

            # --- Preprocessing ---
            xtrain_split_proc = xtrain_split.copy()
            xvalid_split_proc = xvalid_split.copy()
            top_k_features_indices = slice(None) # Default

            if args.method in ["knn", "kmeans"]:
                means = np.mean(xtrain_split_proc, axis=0)
                stds = np.std(xtrain_split_proc, axis=0)
                stds[stds == 0] = 1e-8 # avoid division by zero

                # data preprocessing

                xtrain_split_proc = normalize_fn(xtrain_split_proc, means, stds)
                xvalid_split_proc = normalize_fn(xvalid_split_proc, means, stds)

                xtrain_split_proc = np.clip(xtrain_split_proc, -3, 3)
                xvalid_split_proc = np.clip(xvalid_split_proc, -3, 3)

                min_vals = xtrain_split_proc.min(axis=0)
                max_vals = xtrain_split_proc.max(axis=0)
                range_vals = max_vals - min_vals
                range_vals[range_vals == 0] = 1e-8

                xtrain_split_proc = (xtrain_split_proc - min_vals) / range_vals
                xvalid_split_proc = (xvalid_split_proc - min_vals) / range_vals

                # Rank features by variance on the training split.
                variances = np.var(xtrain_split_proc, axis=0)
                if args.method == "kmeans":
                    if xtrain_split_proc.shape[1] >= 7:
                        top_k_features_indices = np.argsort(variances)[-7:]
                    else: print(f"Warning ({p*100:.0f}% split): Not enough features for KMeans selection (needs 7). Using all.")
                elif args.method == "knn":
                     if xtrain_split_proc.shape[1] >= 10:
                        top_k_features_indices = np.argsort(variances)[-10:]
                     else: print(f"Warning ({p*100:.0f}% split): Not enough features for KNN selection (needs 10). Using all.")

                xtrain_split_proc = xtrain_split_proc[:, top_k_features_indices]
                xvalid_split_proc = xvalid_split_proc[:, top_k_features_indices]

            # --- Initialize Model ---
            method_obj_split = None # initialize to ensure it exists
            if args.method == "dummy_classifier":
                method_obj_split = DummyClassifier(arg1=1, arg2=2)
            elif args.method == "knn":
                 method_obj_split = KNN(k=args.K)
            elif args.method == "kmeans":
                 method_obj_split = KMeans(k=args.K)
            elif args.method == "logistic_regression":
                 # add bias term *after* other preprocessing
                 xtrain_split_proc = append_bias_term(xtrain_split_proc)
                 xvalid_split_proc = append_bias_term(xvalid_split_proc)
                 method_obj_split = LogisticRegression(lr=args.lr, max_iters=args.max_iters)
            else:
                 raise ValueError(f"Unsupported method: {args.method}")

            # --- Train and Evaluate ---
            print(f"Fitting {args.method} model...")
            preds_train_split = method_obj_split.fit(xtrain_split_proc, ytrain_split)
            print("Predicting on validation split...")
            preds_valid_split = method_obj_split.predict(xvalid_split_proc)

            # report results for this split percentage
            acc_train = accuracy_fn(preds_train_split, ytrain_split)
            macrof1_train = macrof1_fn(preds_train_split, ytrain_split)
            print(f"  Train split performance: accuracy = {acc_train:.3f}% - F1-score = {macrof1_train:.6f}")

            acc_valid = accuracy_fn(preds_valid_split, yvalid_split)
            macrof1_valid = macrof1_fn(preds_valid_split, yvalid_split)
            print(f"  Validation split performance: accuracy = {acc_valid:.3f}% - F1-score = {macrof1_valid:.6f}")

            results_per_split[p] = {
                'accuracy': acc_valid,
                'f1_score': macrof1_valid,
                'train_indices': train_indices,
                'valid_indices': valid_indices
            }

        print("\n--- Validation Split Testing Complete ---")
        print("Results summary (Validation Accuracy):")
        for p, metrics in results_per_split.items():
            print(f"  {p*100:.0f}% split: {metrics['accuracy']:.3f}% ({metrics['f1_score']:.6f} F1)")

        # find the best split based on validation accuracy
        best_split = max(results_per_split.items(), key=lambda x: x[1]['accuracy'])[0]
        best_accuracy = results_per_split[best_split]['accuracy']
        print(f"\n--- Best validation split: {best_split*100:.0f}% with accuracy: {best_accuracy:.3f}% ---")

        # get the data for the best split
        best_train_indices = results_per_split[best_split]['train_indices']
        best_valid_indices = results_per_split[best_split]['valid_indices']

        xtrain_best = xtrain_orig[best_train_indices]
        ytrain_best = ytrain_orig[best_train_indices]
        xvalid_best = xtrain_orig[best_valid_indices]
        yvalid_best = ytrain_orig[best_valid_indices]

        xtrain = xtrain_best
        ytrain = ytrain_best
        xtest = xvalid_best
        ytest = yvalid_best

        print(f"Using {best_split*100:.0f}% validation split with best accuracy: {best_accuracy:.3f}%")

    # Normalize data for KNN and KMeans
    if args.method == "knn" or args.method == "kmeans":
        
        # Fit standardization statistics on training data.
        means = np.mean(xtrain, axis=0)
        stds = np.std(xtrain, axis=0)

        # apply normalization
        xtrain = normalize_fn(xtrain, means, stds)
        xtest = normalize_fn(xtest, means, stds)

        # clip values to remove extreme outliers
        med  = np.median(xtrain, axis=0)
        iqr  = np.percentile(xtrain, 75, axis=0) - np.percentile(xtrain, 25, axis=0) + 1e-8
        xtrain = (xtrain - med) / iqr
        xtest  = (xtest  - med) / iqr

        # apply minmax scaling
        min_vals = xtrain.min(axis=0)
        max_vals = xtrain.max(axis=0)
        xtrain = (xtrain - min_vals) / (max_vals - min_vals + 1e-8)
        xtest = (xtest - min_vals) / (max_vals - min_vals + 1e-8)


        # Handle outliers specifically for KMeans
        if args.method == "kmeans":
            variances = np.var(xtrain, axis=0)
            top_k_features = np.argsort(variances)[-7:] 
            xtrain = xtrain[:, top_k_features]
            xtest = xtest[:, top_k_features]
        elif args.method == "knn":
            # for knn, just use top variance features
            variances = np.var(xtrain, axis=0)
            top_k_features = np.argsort(variances)[-10:] 
            xtrain = xtrain[:, top_k_features]
            xtest = xtest[:, top_k_features]

    elif args.method == "logistic_regression":
        
        means = np.mean(xtrain, axis=0)
        stds = np.std(xtrain, axis=0)

        # apply normalization
        xtrain = normalize_fn(xtrain, means, stds)
        xtest = normalize_fn(xtest, means, stds)
        

        xtrain = append_bias_term(xtrain)
        xtest = append_bias_term(xtest)

    ## 3. Initialize the method you want to use.


    # ---------------------------------------------------------------------------------
    #  Cross‑Validation block (skipped if args.cv_folds == 0) done for knn and logreg
    # ---------------------------------------------------------------------------------
    if args.cv_folds and (args.method == "knn" or args.method == "logistic_regression") and args.test == False:
        param_grid = [1, 3, 5, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99]
        build = lambda k: KNN(k=k)
        metric = accuracy_fn                 # or f1
        best_param, cv_table = run_cv(
            xtrain, ytrain, build, metric,
            param_grid, args.cv_folds, seed=42
        )

        print("\nCV results ({}‑fold):".format(args.cv_folds))
        for p, (mean_, std_) in cv_table.items():
            print(f"  param={p:<4}: {mean_:6.3f} ± {std_:5.3f}")

        if args.method in ("knn", "kmeans"):
            args.K = best_param
            print(f"→ Using best {args.method} hyper‑parameter K = {args.K}\n")


    # Neural-network evaluation belongs to the separate image-data component.
    if args.method == "nn":
        raise NotImplementedError("Neural-network models are outside this project's scope.")

    # Construct the requested classifier.
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        method_obj = KNN(k=args.K)

    elif args.method == "kmeans":
        method_obj = KMeans(args.K)
        
    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)

    ## 4. Train and evaluate the method
    # Fit (:=train) the method on the training data for classification task
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    # Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")



if __name__ == "__main__":
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / knn / logistic_regression / kmeans",
    )
    parser.add_argument(
        "--data_path", default="features.npz", type=str, help="path to your dataset"
    )
    parser.add_argument(
        "--data_type", default="features", type=str, help="features/original(MS2)"
    )
    parser.add_argument(
        "--K", type=int, default=1, help="number of neighboring datapoints used for knn"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, otherwise use a validation set",
    )


    parser.add_argument(
    "--cv_folds",
    type=int,
    default=0,                  # 0 ⇒ no CV; 5 ⇒ 5‑fold CV
    help="Number of folds for cross‑validation (0 disables CV)",
)


    # MS2 arguments
    parser.add_argument(
        "--nn_type",
        default="cnn",
        help="which network to use, can be 'Transformer' or 'cnn'",
    )
    parser.add_argument(
        "--nn_batch_size", type=int, default=64, help="batch size for NN training"
    )

    # KMeans-specific arguments
    parser.add_argument(
        "--use_pca", action="store_true", help="use PCA for dimensionality reduction (KMeans)"
    )
    parser.add_argument(
        "--pca_components", type=int, default=5, help="number of PCA components to use"
    )

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
