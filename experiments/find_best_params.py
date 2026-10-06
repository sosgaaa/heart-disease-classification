import numpy as np
import argparse
from src.methods.knn import KNN
from src.utils import normalize_fn, accuracy_fn, macrof1_fn

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='features.npz', type=str, help='path to your dataset')
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    feature_data = np.load(args.data_path, allow_pickle=True)
    xtrain_full, xtest = feature_data["xtrain"], feature_data["xtest"]
    ytrain_full, ytest = feature_data["ytrain"], feature_data["ytest"]
    
    # Create validation split (use 20% of training data for validation)
    np.random.seed(42)
    val_size = int(0.2 * xtrain_full.shape[0])
    indices = np.random.permutation(xtrain_full.shape[0])
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    xtrain = xtrain_full[train_indices]
    ytrain = ytrain_full[train_indices]
    xval = xtrain_full[val_indices]
    yval = ytrain_full[val_indices]
    
    print(f"Training set: {xtrain.shape[0]} samples")
    print(f"Validation set: {xval.shape[0]} samples")
    
    # Preprocessing
    print("Preprocessing data...")
    
    # Implement quadratic feature expansion
    print("Applying quadratic feature expansion...")
    
    def quadratic_expansion(X):
        """
        Expand features by adding quadratic terms (squares and interactions)
        """
        n_samples, n_features = X.shape
        
        # Initialize expanded feature matrix with original features
        X_expanded = X.copy()
        
        # Add squares of each feature
        squares = X**2
        X_expanded = np.hstack((X_expanded, squares))
        
        # Add cross-products between features
        for i in range(n_features):
            for j in range(i+1, n_features):
                cross_term = (X[:, i] * X[:, j]).reshape(-1, 1)
                X_expanded = np.hstack((X_expanded, cross_term))
        
        print(f"Features expanded from {n_features} to {X_expanded.shape[1]}")
        return X_expanded
    
    # Apply feature expansion
    xtrain = quadratic_expansion(xtrain)
    xval = quadratic_expansion(xval)
    xtest = quadratic_expansion(xtest)
    
    # Normalize data
    means = np.mean(xtrain, axis=0)
    stds = np.std(xtrain, axis=0)
    xtrain_norm = normalize_fn(xtrain, means, stds)
    xval_norm = normalize_fn(xval, means, stds)
    
    # Parameters to search
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    top_features_values = [5, 7, 10, 13, 'all']
    
    best_accuracy = 0
    best_f1 = 0
    best_k = 1
    best_features = 'all'
    best_feature_indices = np.arange(xtrain.shape[1])
    
    results = []
    
    # Grid search
    for k in k_values:
        for n_features in top_features_values:
            print(f"\nTrying k={k}, features={n_features}")
            
            # Feature selection
            if n_features == 'all':
                feature_indices = np.arange(xtrain_norm.shape[1])
            else:
                # Select top features by variance
                variances = np.var(xtrain_norm, axis=0)
                feature_indices = np.argsort(variances)[-n_features:]
            
            # Select features
            xtrain_selected = xtrain_norm[:, feature_indices]
            xval_selected = xval_norm[:, feature_indices]
            
            # Train KNN
            knn = KNN(k=k)
            preds_train = knn.fit(xtrain_selected, ytrain)
            preds_val = knn.predict(xval_selected)
            
            # Evaluate
            accuracy = accuracy_fn(preds_val, yval)
            f1_score = macrof1_fn(preds_val, yval)
            
            results.append((k, n_features, accuracy, f1_score))
            print(f"Validation accuracy: {accuracy:.2f}%, F1 score: {f1_score:.4f}")
            
            # Track best model by accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
                best_features = n_features
                best_feature_indices = feature_indices
            
            # Track best model by F1 score
            if f1_score > best_f1:
                best_f1 = f1_score
                
    # Print results sorted by accuracy
    print("\n===== Results sorted by accuracy =====")
    for k, features, acc, f1 in sorted(results, key=lambda x: x[2], reverse=True)[:5]:
        print(f"k={k}, features={features}: Accuracy={acc:.2f}%, F1={f1:.4f}")
    
    # Print results sorted by F1 score
    print("\n===== Results sorted by F1 score =====")
    for k, features, acc, f1 in sorted(results, key=lambda x: x[3], reverse=True)[:5]:
        print(f"k={k}, features={features}: Accuracy={acc:.2f}%, F1={f1:.4f}")
    
    # Final evaluation on test set with best params
    print("\n===== Final evaluation on test set =====")
    xtest_norm = normalize_fn(xtest, means, stds)
    xtest_selected = xtest_norm[:, best_feature_indices]
    
    knn = KNN(k=best_k)
    knn.fit(xtrain_selected, ytrain)
    preds_test = knn.predict(xtest_selected)
    
    test_accuracy = accuracy_fn(preds_test, ytest)
    test_f1 = macrof1_fn(preds_test, ytest)
    
    print(f"Best parameters: k={best_k}, features={best_features}")
    print(f"Test accuracy: {test_accuracy:.2f}%, F1 score: {test_f1:.4f}")
    
    print("\nTo use these parameters:")
    print(f"python main.py --method knn --K {best_k}")

if __name__ == "__main__":
    main() 