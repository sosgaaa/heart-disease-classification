import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from src.methods.logistic_regression import LogisticRegression
from src.utils import accuracy_fn, normalize_fn, append_bias_term


def test_logistic_params(xtrain, ytrain, xtest, ytest, lr_values, max_iters_values):
    """
    Test logistic regression with different combinations of parameters
    and return the results.
    """
    # Normalize data
    means = np.mean(xtrain, axis=0)
    stds = np.std(xtrain, axis=0)
    xtrain = normalize_fn(xtrain, means, stds)
    xtest = normalize_fn(xtest, means, stds)
    xtrain = append_bias_term(xtrain)
    xtest = append_bias_term(xtest)

    
    results = {
        'test_accuracy': np.zeros((len(lr_values), len(max_iters_values)))
    }
    
    for i, lr in enumerate(lr_values):
        for j, max_iters in enumerate(max_iters_values):
            print(f"Testing lr={lr:.8f}, max_iters={max_iters}")
            
            # Initialize and train model
            model = LogisticRegression(lr=lr, max_iters=max_iters)
            model.fit(xtrain, ytrain)
            preds_test = model.predict(xtest)
            
            # Calculate accuracy
            results['test_accuracy'][i,j] = accuracy_fn(preds_test, ytest)
    
    return results

def plot_results(lr_values, max_iters_values, results):
    """
    Create 3D plot of Accuracy surface.
    """
    # Find best parameters
    best_idx = np.unravel_index(np.argmax(results['test_accuracy']), results['test_accuracy'].shape)
    best_lr = lr_values[best_idx[0]]
    best_max_iters = max_iters_values[best_idx[1]]
    best_accuracy = results['test_accuracy'][best_idx]
    
    # 3D Surface Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for surface plot
    X, Y = np.meshgrid(np.log10(lr_values), max_iters_values)
    Z = results['test_accuracy'].T
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Plot best point
    best_x = np.log10(best_lr)
    best_y = best_max_iters
    best_z = best_accuracy
    ax.scatter(best_x, best_y, best_z, color='red', s=100)
    
    # Add projection lines
    z_min = ax.get_zlim()[0] # Get the bottom limit of the z-axis for the projection line
    ax.plot([best_x, best_x], [best_y, best_y], [z_min, best_z], 'r--', alpha=0.5) # Vertical line to floor (use z_min)
    ax.plot([ax.get_xlim()[0], best_x], [best_y, best_y], [best_z, best_z], 'r--', alpha=0.5) # Line to YZ plane (back wall)
    ax.plot([best_x, best_x], [ax.get_ylim()[0], best_y], [best_z, best_z], 'r--', alpha=0.5) # Line to XZ plane (side wall)

    # Add text labels for the best point coordinates near the projection endpoints
    ax.text(best_x, best_y, z_min, f'Iter={best_max_iters}', color='red', ha='center', va='top') # Label on the 'floor' (Y value)
    ax.text(ax.get_xlim()[0], best_y, best_z, f'Acc={best_accuracy:.3f}', color='red', ha='left', va='center') # Label on the back wall (Z value)
    ax.text(best_x, ax.get_ylim()[0], best_z, f'logLR={best_x:.1f}', color='red', ha='center', va='bottom') # Label on the side wall (X value)

    # Customize the view
    ax.view_init(elev=20, azim=45)
    ax.set_xlabel('Log10(Learning Rate)')
    ax.set_ylabel('Max Iterations')
    ax.set_zlabel('Accuracy')
    ax.set_title('Accuracy Surface')
    ax.set_xlim(np.log10(lr_values).min(),
            np.log10(lr_values).max())
    ax.set_ylim(max_iters_values.min(),
                max_iters_values.max())
    ax.set_zlim(results['test_accuracy'].min(),
                results['test_accuracy'].max())

    plt.savefig('logistic_accuracy_surface.png')
    plt.close()
    
    # Print best parameters
    print(f"\nBest parameters found:")
    print(f"Learning Rate: {best_lr:.1e}")
    print(f"Max Iterations: {best_max_iters}")
    print(f"Test Accuracy: {best_accuracy:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='Data_MS1_2025/features.npz', type=str)
    args = parser.parse_args()
    
    # Load data
    feature_data = np.load(args.data_path, allow_pickle=True)
    xtrain, xtest = feature_data["xtrain"], feature_data["xtest"]
    ytrain, ytest = feature_data["ytrain"], feature_data["ytest"]


    # Method-specific processing for logistic regression
    means = np.mean(xtrain, axis=0)
    stds = np.std(xtrain, axis=0)
    xtrain = normalize_fn(xtrain, means, stds)
    xtest = normalize_fn(xtest, means, stds)

    # Test different parameters
    lr_values = np.array([10**i for i in range(-8, -1)])
    max_iters_values = np.arange(125, 3000, 125, dtype=int)
    results = test_logistic_params(xtrain, ytrain, xtest, ytest, lr_values, max_iters_values)
    plot_results(lr_values, max_iters_values, results)

if __name__ == "__main__":
    main() 