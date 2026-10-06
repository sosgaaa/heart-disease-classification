import numpy as np
import matplotlib.pyplot as plt
from src.methods.knn import KNN
from src.utils import accuracy_fn, normalize_fn, macrof1_fn
import argparse

def plot_knn_error(xtrain, ytrain, xtest, ytest, k_values):
    """
    Plot the error rate of KNN classifier for different values of k.
    
    Arguments:
        xtrain, ytrain: Training data and labels
        xtest, ytest: Test data and labels
        k_values: List of k values to try
        num_features: Number of top features to use
    """   
   
    # 2. Normalize data
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

    
    # 3. Feature selection
    variances = np.var(xtrain, axis=0)
    top_k_features = np.argsort(variances)[-10:]
    xtrain = xtrain[:, top_k_features]
    xtest = xtest[:, top_k_features]
    
    train_errors = []
    test_errors = []
    train_f1s = []
    test_f1s = []
    
    for k in k_values:
        # Initialize KNN with current k
        knn = KNN(k=k)
        
        # Train and get predictions
        preds_train = knn.fit(xtrain, ytrain)
        preds_test = knn.predict(xtest)
        
        # Calculate error rates (1 - accuracy)
        train_acc = accuracy_fn(preds_train, ytrain)
        test_acc = accuracy_fn(preds_test, ytest)
        
        # Calculate F1 scores
        train_f1 = macrof1_fn(preds_train, ytrain)
        test_f1 = macrof1_fn(preds_test, ytest)
        
        train_errors.append(100 - train_acc)  # Convert to error percentage
        test_errors.append(100 - test_acc)
        train_f1s.append(train_f1)
        test_f1s.append(test_f1)
        
        print(f"K={k}: Train error = {100-train_acc:.2f}%, Test error = {100-test_acc:.2f}%, F1 = {test_f1:.4f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Error rate plot
    ax1.plot(k_values, train_errors, 'o-', label='Training Error')
    ax1.plot(k_values, test_errors, 's-', label='Test Error')
    ax1.set_xlabel('Number of Neighbors (k)')
    ax1.set_ylabel('Error Rate (%)')
    ax1.set_title('KNN Error Rate vs. k')
    ax1.legend()
    ax1.grid(True)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.set_xticks(k_values)  # Set x-ticks to show only odd k values
    
    # F1 score plot
    ax2.plot(k_values, train_f1s, 'o-', label='Training F1')
    ax2.plot(k_values, test_f1s, 's-', label='Test F1')
    ax2.set_xlabel('Number of Neighbors (k)')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('KNN F1 Score vs. k')
    ax2.legend()
    ax2.grid(True)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.set_xticks(k_values)  # Set x-ticks to show only odd k values
    
    plt.tight_layout()
    plt.savefig('knn_performance_vs_k.png')
    plt.show()
    
    # Find k with minimum test error and maximum F1
    best_k_error = k_values[np.argmin(test_errors)]
    best_k_f1 = k_values[np.argmax(test_f1s)]
    print(f"\nBest k value (minimum error): {best_k_error} with test error: {min(test_errors):.2f}%")
    print(f"Best k value (maximum F1): {best_k_f1} with test F1: {max(test_f1s):.4f}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_k', type=int, default=1, help='minimum k value')
    parser.add_argument('--max_k', type=int, default=21, help='maximum k value')
    parser.add_argument('--step', type=int, default=2, help='step size for k values')  # Changed to 2 to get odd numbers
    args = parser.parse_args()
    
    # Load data
    feature_data = np.load("features.npz", allow_pickle=True)
    xtrain, xtest = feature_data["xtrain"], feature_data["xtest"]
    ytrain, ytest = feature_data["ytrain"], feature_data["ytest"]
    
    # Generate k values to try - only odd values from 1 to 21
    k_values = list(range(args.min_k, args.max_k + 1, args.step))
    
    # Plot error rate vs k
    plot_knn_error(xtrain, ytrain, xtest, ytest, k_values)

if __name__ == "__main__":
    main() 