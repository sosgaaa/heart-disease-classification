import numpy as np
import matplotlib.pyplot as plt
# Assuming kmeans.py containing the KMeans class is in the same directory
# or accessible via the Python path.
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, accuracy_fn # Import accuracy_fn directly

# Data loading with the preprocessing used by the experiment scripts.
def load_data(file_path="features.npz"):
    """
    Loads and preprocesses data from the specified .npz file,
    applying Z-score normalization, Median/IQR scaling, and MinMax scaling (0, 1).
    Preprocessing steps are fitted on the training data only.

    Args:
        file_path (str): Path to the features.npz file.

    Returns:
        tuple: (xtrain, ytrain, xtest, ytest) preprocessed data.
               Returns dummy data if the file is not found or keys are missing.
    """
    try:
        print(f"Loading data from: {file_path}")
        feature_data = np.load(file_path, allow_pickle=True)
        xtrain, xtest = feature_data["xtrain"], feature_data["xtest"]
        ytrain, ytest = feature_data["ytrain"], feature_data["ytest"]
        print(f"Data loaded. xtrain shape: {xtrain.shape}, xtest shape: {xtest.shape}")

        # Ensure data has the expected number of features (optional but good practice)
        if xtrain.shape[1] != 13 or xtest.shape[1] != 13:
             print(f"Warning: Loaded data does not have 13 features (Train: {xtrain.shape[1]}, Test: {xtest.shape[1]}). Proceeding anyway.")

        print("Preprocessing data (Normalize -> Median/IQR Scale -> MinMax)...")
        # 1. Z-score Normalization (using training stats)
        means = np.mean(xtrain, axis=0)
        stds = np.std(xtrain, axis=0)
        # Use provided normalize_fn if it performs (data - mean) / std
        xtrain = normalize_fn(xtrain, means, stds)
        xtest = normalize_fn(xtest, means, stds) # Use training stats for test set

        # 2. Median/IQR Scaling (using training stats)
        med = np.median(xtrain, axis=0)
        iqr = np.percentile(xtrain, 75, axis=0) - np.percentile(xtrain, 25, axis=0) + 1e-8 # Add epsilon for stability
        xtrain = (xtrain - med) / iqr
        xtest = (xtest - med) / iqr # Use training stats for test set

        # 3. Min-Max Scaling (using training stats after Median/IQR scaling)
        min_vals = xtrain.min(axis=0)
        max_vals = xtrain.max(axis=0)
        # Add epsilon to prevent division by zero if a feature has zero range after scaling
        range_vals = max_vals - min_vals + 1e-8
        xtrain = (xtrain - min_vals) / range_vals
        xtest = (xtest - min_vals) / range_vals # Use training stats for test set

        print("Preprocessing complete.")
        return xtrain, ytrain, xtest, ytest

    except FileNotFoundError:
        print(f"Error: Data file not found at '{file_path}'")
        print("Please ensure 'features.npz' is in the correct directory relative to the script.")
        # Fallback to prevent crashing, but highlight the error
        print("CRITICAL: Returning dummy data as fallback. Results will be meaningless.")
        n_samples_train = 100; n_samples_test = 20; n_features = 13; n_classes = 10
        return (np.random.rand(n_samples_train, n_features),
                np.random.randint(0, n_classes, n_samples_train),
                np.random.rand(n_samples_test, n_features),
                np.random.randint(0, n_classes, n_samples_test))
    except KeyError as e:
         print(f"Error: Missing key {e} in '{file_path}'. Expected 'xtrain', 'ytrain', 'xtest', 'ytest'.")
         # Fallback
         print("CRITICAL: Returning dummy data as fallback. Results will be meaningless.")
         n_samples_train = 100; n_samples_test = 20; n_features = 13; n_classes = 10
         return (np.random.rand(n_samples_train, n_features),
                 np.random.randint(0, n_classes, n_samples_train),
                 np.random.rand(n_samples_test, n_features),
                 np.random.randint(0, n_classes, n_samples_test))

# calculate_accuracy remains the same
def calculate_accuracy(y_pred, y_true):
    """Calculates the accuracy of predictions."""
    if len(y_pred) != len(y_true):
        raise ValueError("Prediction and true label arrays must have the same length.")
    # Use accuracy_fn from utils if it returns percentage, otherwise multiply by 100
    # Assuming accuracy_fn returns percentage based on its usage elsewhere
    return accuracy_fn(y_pred, y_true) # Return accuracy as percentage


def evaluate_kmeans_feature_and_cluster_sweep(x_train, y_train, x_test, y_test,
                                              num_features_to_evaluate,
                                              max_k_clusters=100, random_state=42):
    """
    Evaluates KMeans accuracy by varying the number of top variance features
    and finding the best number of clusters (KMeans k) for each feature count.
    Uses data preprocessed according to load_data function.

    Args:
        x_train (np.array): Preprocessed training features.
        y_train (np.array): Training labels.
        x_test (np.array): Preprocessed testing features.
        y_test (np.array): Testing labels.
        num_features_to_evaluate (int): Max number of top features (k_feat) to evaluate.
        max_k_clusters (int): Maximum number of clusters (KMeans k) to test for each feature count.
        random_state (int): Random seed for KMeans reproducibility.

    Returns:
        tuple: (list: number of features used [1, ..., num_features_to_evaluate],
                list: *best* accuracy (%) found for each feature count,
                list: best number of clusters (KMeans k) found for each feature count,
                list: indices of features sorted by variance)
    """
    n_total_features = x_train.shape[1]
    if num_features_to_evaluate > n_total_features:
        print(f"Warning: Requested to evaluate {num_features_to_evaluate} features, but only {n_total_features} available. Evaluating all.")
        num_features_to_evaluate = n_total_features
    elif num_features_to_evaluate <= 0:
         raise ValueError("num_features_to_evaluate must be positive.")

    # Calculate variance for each feature on the *preprocessed* training data
    variances = np.var(x_train, axis=0)
    sorted_feature_indices = np.argsort(variances)[::-1] # Descending order

    best_accuracies_per_feature_count = []
    best_k_clusters_per_feature_count = []
    num_features_list = list(range(1, num_features_to_evaluate + 1))
    k_clusters_range = list(range(1, max_k_clusters + 1))

    print(f"Evaluating KMeans: Sweeping top features (1 to {num_features_to_evaluate}) and KMeans clusters (1 to {max_k_clusters})...")

    # Outer loop: Number of features (k_feat)
    for k_feat in num_features_list:
        current_feature_indices = sorted_feature_indices[:k_feat]
        x_train_subset = x_train[:, current_feature_indices]
        x_test_subset = x_test[:, current_feature_indices]

        current_best_acc = -1.0
        current_best_k_clusters = -1

        print(f"  Testing with top {k_feat} features:")

        # Inner loop: Number of clusters for KMeans (k_clusters)
        for k_clusters in k_clusters_range:
            # Initialize and train KMeans
            kmeans = KMeans(k=k_clusters, random_state=random_state) # Use the passed random_state
            try:
                kmeans.fit(x_train_subset, y_train)
                y_pred = kmeans.predict(x_test_subset)
                accuracy = calculate_accuracy(y_pred, y_test)

                if accuracy > current_best_acc:
                    current_best_acc = accuracy
                    current_best_k_clusters = k_clusters

            except Exception as e:
                # Handle potential errors during KMeans
                print(f"    Error during KMeans fit/predict for k_feat={k_feat}, k_clusters={k_clusters}: {e}")

        # Store the actual best results found for this k_feat
        best_accuracies_per_feature_count.append(current_best_acc)
        best_k_clusters_per_feature_count.append(current_best_k_clusters)

        print(f"  => Best accuracy for top {k_feat} features: {current_best_acc:.2f}% (found with k_clusters={current_best_k_clusters})")

    return num_features_list, best_accuracies_per_feature_count, best_k_clusters_per_feature_count, sorted_feature_indices.tolist()


# plot_accuracy_vs_features remains the same
def plot_accuracy_vs_features(num_features_list, best_accuracies, best_k_clusters_list):
    """Plots the best accuracy vs. the number of top variance features used."""
    if not num_features_list or not best_accuracies:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(num_features_list, best_accuracies, marker='o', linestyle='-', label='Best Accuracy per Feature Count')

    if best_accuracies:
        # Find index of max accuracy, handling potential -1 values if errors occurred
        valid_accuracies = [acc for acc in best_accuracies if acc >= 0]
        if valid_accuracies:
            max_overall_accuracy = max(valid_accuracies)
            best_feature_index_in_list = best_accuracies.index(max_overall_accuracy) # Find first occurrence
            best_num_features = num_features_list[best_feature_index_in_list]
            best_k_for_best_features = best_k_clusters_list[best_feature_index_in_list]

            plt.scatter(best_num_features, max_overall_accuracy, color='red', s=100, zorder=5,
                        label=f'Overall Best: {max_overall_accuracy:.2f}% \n(at {best_num_features} features, k_clusters={best_k_for_best_features})')
            plt.axvline(x=best_num_features, color='red', linestyle='--', linewidth=1)
        else:
            print("Warning: No valid accuracy results found to determine the overall best.")


    plt.xlabel("Number of Top Variance Features Used")
    plt.ylabel("Best KMeans Accuracy (%)")
    # Update title to reflect the range actually tested
    tested_k_range = f"1-{len(best_k_clusters_list)}" if best_k_clusters_list else "N/A"
    plt.title(f"Best KMeans Accuracy vs. Number of Top Variance Features (Optimized over k_clusters {tested_k_range})")
    min_acc = min(valid_accuracies) if valid_accuracies else 0
    max_acc = max(valid_accuracies) if valid_accuracies else 100
    plt.ylim(max(0, min_acc - 5), min(100, max_acc + 5))
    plt.xlim(0.5, num_features_list[-1] + 0.5 if num_features_list else 10)
    plt.xticks(num_features_list)
    plt.grid(True, linestyle='-', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("kmeans_best_accuracy_vs_features_clusters_sweep.png")
    print("\nPlot saved as kmeans_best_accuracy_vs_features_clusters_sweep.png")
    plt.show()


# Main function remains the same, including the verification step
def main():
    # --- Configuration ---
    NUM_FEATURES_TO_EVALUATE = 13
    MAX_K_CLUSTERS_TO_TEST = 80 # Set the range for KMeans k parameter
    RANDOM_SEED = 42
    DATA_FILE_PATH = "features.npz"
    # -------------------

    # Load and preprocess data once.
    x_train, y_train, x_test, y_test = load_data(DATA_FILE_PATH)

    # Validation (remains the same)
    if not all(isinstance(arr, np.ndarray) for arr in [x_train, y_train, x_test, y_test]):
        print("Error: Data loading did not return numpy arrays. Exiting.")
        return
    if x_train.shape[0] == 0 or x_test.shape[0] == 0:
        print("Error: Loaded data appears empty. Exiting.")
        return
    if x_train.shape[0] != y_train.shape[0] or x_test.shape[0] != y_test.shape[0]:
        print("Error: Mismatch between number of samples and labels. Exiting.")
        return
    if x_train.shape[1] != x_test.shape[1]:
        print("Error: Mismatch in number of features between train and test sets. Exiting.")
        return

    actual_features = x_train.shape[1]
    if actual_features < NUM_FEATURES_TO_EVALUATE:
         print(f"Warning: Dataset has only {actual_features} features. Adjusting evaluation range to 1-{actual_features}.")
         num_to_eval = actual_features
    elif actual_features > NUM_FEATURES_TO_EVALUATE:
         print(f"Warning: Dataset has {actual_features} features, evaluating top 1 to {NUM_FEATURES_TO_EVALUATE}.")
         num_to_eval = NUM_FEATURES_TO_EVALUATE
    else:
        num_to_eval = NUM_FEATURES_TO_EVALUATE

    # Run the main evaluation function
    num_features_list, best_accuracies, best_k_clusters_list, sorted_indices = evaluate_kmeans_feature_and_cluster_sweep(
        x_train, y_train, x_test, y_test,
        num_features_to_evaluate=num_to_eval,
        max_k_clusters=MAX_K_CLUSTERS_TO_TEST,
        random_state=RANDOM_SEED
    )

    # Find best results, handling potential -1 values
    valid_indices = [i for i, acc in enumerate(best_accuracies) if acc >= 0]

    if valid_indices:
        best_feature_index_in_list = valid_indices[np.argmax([best_accuracies[i] for i in valid_indices])]
        max_overall_accuracy = best_accuracies[best_feature_index_in_list]
        best_num_features = num_features_list[best_feature_index_in_list]
        best_k_for_best_features = best_k_clusters_list[best_feature_index_in_list]

        print(f"\n--- Results Summary ---")
        print(f"Evaluated using top k features (k=1 to {len(num_features_list)}) sorted by variance.")
        print(f"For each feature count, tested KMeans k_clusters from 1 to {MAX_K_CLUSTERS_TO_TEST}.")
        print(f"Feature indices sorted by variance (desc): {sorted_indices}")
        print("-" * 20)
        for i, k_feat in enumerate(num_features_list):
            acc_str = f"{best_accuracies[i]:.2f}%" if best_accuracies[i] >= 0 else "Error"
            k_clus_str = str(best_k_clusters_list[i]) if best_k_clusters_list[i] >= 0 else "N/A"
            print(f"Top {k_feat} Features: Best Accuracy = {acc_str} (with k_clusters = {k_clus_str})")
        print("-" * 20)
        print(f"Overall Maximum accuracy achieved: {max_overall_accuracy:.2f}%")
        print(f"Optimal number of features: {best_num_features}")
        print(f"Optimal number of clusters (k_clusters) for that feature count: {best_k_for_best_features}")
        print(f"---------------")

        # --- Verification Step ---
        print("\n--- Verifying Best Result Manually ---")
        print(f"Re-running KMeans with best parameters found:")
        print(f"  Number of top features = {best_num_features}")
        print(f"  Number of clusters (k) = {best_k_for_best_features}")
        print(f"  Random State = {RANDOM_SEED}")

        # 1. Select the exact same top features
        verification_feature_indices = sorted_indices[:best_num_features]
        x_train_verify_subset = x_train[:, verification_feature_indices]
        x_test_verify_subset = x_test[:, verification_feature_indices]
        print(f"  Using feature indices: {verification_feature_indices}")

        # 2. Initialize KMeans with the exact same parameters
        verification_kmeans = KMeans(k=best_k_for_best_features, random_state=RANDOM_SEED)

        # 3. Fit and Predict
        try:
            verification_kmeans.fit(x_train_verify_subset, y_train)
            verification_preds = verification_kmeans.predict(x_test_verify_subset)

            # 4. Calculate Accuracy
            verification_accuracy = calculate_accuracy(verification_preds, y_test)
            print(f"  Accuracy calculated in verification: {verification_accuracy:.2f}%")

            # 5. Compare
            if np.isclose(verification_accuracy, max_overall_accuracy):
                print("  Verification successful: Accuracy matches the best accuracy found in the sweep.")
            else:
                print(f"  Verification FAILED: Accuracy ({verification_accuracy:.2f}%) does NOT match best accuracy from sweep ({max_overall_accuracy:.2f}%).")
                print("  Check for inconsistencies in data handling or model parameters between the loop and verification.")

        except Exception as e:
            print(f"  Error during verification run: {e}")
        print("------------------------------------")


        # Plot results
        plot_accuracy_vs_features(num_features_list, best_accuracies, best_k_clusters_list)
    else:
        print("\nEvaluation could not be completed or yielded no valid results.")

if __name__ == "__main__":
    main()