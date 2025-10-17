import numpy as np
from typing import Tuple
from tree_node import TreeNode
from decision_tree_builder import decision_tree_learning

def load_datasets() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the clean and noisy WiFi datasets.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (clean_data, noisy_data) where each dataset is a 2000x8 array
               - First 7 columns: WiFi signal strengths (continuous features)
               - Last column: Room number (label)
    """
    # Load clean dataset
    clean_data: np.ndarray = np.loadtxt('wifi_db/clean_dataset.txt')
    
    # Load noisy dataset  
    noisy_data: np.ndarray = np.loadtxt('wifi_db/noisy_dataset.txt')
    
    return clean_data, noisy_data

def extract_features_labels(dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features and labels from a dataset.
    
    Args:
        dataset: Dataset array with features in first 7 columns and labels in last column
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (features, labels)
    """
    features: np.ndarray = dataset[:, :-1]  # First 7 columns (WiFi signals)
    labels: np.ndarray = dataset[:, -1]     # Last column (room numbers)
    return features, labels

if __name__ == "__main__":
    # Load the datasets
    clean_dataset: np.ndarray
    noisy_dataset: np.ndarray
    clean_dataset, noisy_dataset = load_datasets()
    
    print(f"Clean dataset shape: {clean_dataset.shape}")
    print(f"Noisy dataset shape: {noisy_dataset.shape}")
    
    # Test decision tree learning
    print("\n" + "="*50)
    print("Building Decision Tree on Clean Dataset")
    print("="*50)
    
    tree: TreeNode = decision_tree_learning(clean_dataset, depth=0)
    
    print(f"Tree built successfully!")
    print(f"Tree depth: {tree.get_depth()}")
    print(f"Number of leaves: {tree.get_leaf_count()}")
    
    print(f"\n" + "="*60)
    print("DECISION TREE STRUCTURE")
    print("="*60)
    tree.print_tree()
    
    # Test prediction on a few samples
    print(f"\n" + "="*60)
    print("PREDICTION TEST")
    print("="*60)
    
    test_samples: np.ndarray = clean_dataset[:5, :-1]  # First 5 samples, features only
    actual_labels: np.ndarray = clean_dataset[:5, -1]  # First 5 samples, labels only
    
    print(f"Testing on first 5 samples:")
    for i in range(5):
        prediction: int = tree.predict(test_samples[i])
        actual: int = int(actual_labels[i])
        print(f"Sample {i+1}: Predicted Room {prediction}, Actual Room {actual}")
    
    accuracy: float = np.mean([tree.predict(test_samples[i]) == int(actual_labels[i]) for i in range(5)])
    print(f"Accuracy on test samples: {accuracy:.2f}")
    
    # Compare clean vs noisy datasets
    print(f"\n" + "="*60)
    print("CLEAN vs NOISY DATASET COMPARISON")
    print("="*60)
    
    # Build tree on noisy dataset
    noisy_tree: TreeNode = decision_tree_learning(noisy_dataset, depth=0)
    
    print(f"Clean dataset tree depth: {tree.get_depth()}")
    print(f"Noisy dataset tree depth: {noisy_tree.get_depth()}")
    print(f"Clean dataset tree leaves: {tree.get_leaf_count()}")
    print(f"Noisy dataset tree leaves: {noisy_tree.get_leaf_count()}")
    
    # Test accuracy on both datasets
    clean_features, clean_labels = extract_features_labels(clean_dataset)
    noisy_features, noisy_labels = extract_features_labels(noisy_dataset)
    
    # Test clean tree on clean data
    clean_predictions = tree.predict_batch(clean_features[:100])  # Test on first 100 samples
    clean_accuracy = np.mean(clean_predictions == clean_labels[:100])
    
    # Test noisy tree on noisy data
    noisy_predictions = noisy_tree.predict_batch(noisy_features[:100])  # Test on first 100 samples
    noisy_accuracy = np.mean(noisy_predictions == noisy_labels[:100])
    
    print(f"Clean tree accuracy on clean data: {clean_accuracy:.3f}")
    print(f"Noisy tree accuracy on noisy data: {noisy_accuracy:.3f}")