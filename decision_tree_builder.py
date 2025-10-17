import numpy as np
from typing import Tuple
from tree_node import TreeNode

def all_labels_same(dataset: np.ndarray) -> bool:
    """
    Check if all labels in the dataset are the same.
    
    Args:
        dataset: Dataset array with labels in the last column
        
    Returns:
        bool: True if all labels are the same, False otherwise
    """
    labels: np.ndarray = dataset[:, -1]  # Get the last column (labels)
    unique_labels: np.ndarray = np.unique(labels)
    return len(unique_labels) == 1

def decision_tree_learning(dataset: np.ndarray, depth: int = 0) -> TreeNode:
    """
    Build a decision tree using the ID3 algorithm.
    
    Args:
        dataset: Dataset array with features in first n-1 columns and labels in last column
        depth: Current depth in the tree
        
    Returns:
        TreeNode: Root node of the constructed decision tree
    """
    if all_labels_same(dataset):
        return TreeNode(is_leaf=True, label=dataset[0, -1], depth=depth)
    
    split_value = find_split_point(dataset)
    left_dataset: np.ndarray = dataset[dataset[:, split_value[0]] <= split_value[1]]
    right_dataset: np.ndarray = dataset[dataset[:, split_value[0]] > split_value[1]]
    
    left_child: TreeNode = decision_tree_learning(left_dataset, depth + 1)
    right_child: TreeNode = decision_tree_learning(right_dataset, depth + 1)
    
    return TreeNode(
        is_leaf=False, 
        feature_index=split_value[0], 
        threshold=split_value[1], 
        depth=depth,
        left_child=left_child, 
        right_child=right_child
    )

def find_split_point(dataset: np.ndarray) -> Tuple[int, float]:
    """
    Find the best split point for a dataset using information gain.
    
    Args:
        dataset: Dataset array with features in first n-1 columns and labels in last column
        
    Returns:
        Tuple[int, float]: (best_feature_index, best_threshold)
    """
    best_gain: float = 0.0
    best_feature: int = 0
    best_threshold: float = 0.0
    
    n_features: int = dataset.shape[1] - 1  # Exclude label column
    labels: np.ndarray = dataset[:, -1]
    
    # Calculate parent entropy
    parent_entropy: float = calculate_entropy(labels)
    
    for feature_idx in range(n_features):
        feature_values: np.ndarray = dataset[:, feature_idx]
        
        # Sort indices by feature values
        sorted_indices: np.ndarray = np.argsort(feature_values)
        sorted_values: np.ndarray = feature_values[sorted_indices]
        sorted_labels: np.ndarray = labels[sorted_indices]
        
        # Only consider split points between consecutive different values
        for i in range(len(sorted_values) - 1):
            if sorted_values[i] != sorted_values[i + 1]:
                threshold: float = (sorted_values[i] + sorted_values[i + 1]) / 2
                
                # Calculate information gain for this split
                gain: float = calculate_information_gain(dataset, feature_idx, threshold, parent_entropy)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
    
    return best_feature, best_threshold

def calculate_entropy(labels: np.ndarray) -> float:
    """
    Calculate entropy of a set of labels.
    
    Args:
        labels: Array of class labels
        
    Returns:
        float: Entropy value
    """
    if len(labels) == 0:
        return 0.0
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    
    # Avoid log(0) by filtering out zero probabilities
    probabilities = probabilities[probabilities > 0]
    
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_information_gain(dataset: np.ndarray, feature_idx: int, threshold: float, parent_entropy: float) -> float:
    """
    Calculate information gain for a specific split.
    
    Args:
        dataset: Dataset array
        feature_idx: Index of feature to split on
        threshold: Threshold value for splitting
        parent_entropy: Entropy of parent node
        
    Returns:
        float: Information gain
    """
    feature_values: np.ndarray = dataset[:, feature_idx]
    labels: np.ndarray = dataset[:, -1]
    
    # Split the data
    left_mask: np.ndarray = feature_values <= threshold
    right_mask: np.ndarray = ~left_mask
    
    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        return 0.0  # No gain if one side is empty
    
    # Calculate weighted entropy of children
    n_total: int = len(labels)
    n_left: int = np.sum(left_mask)
    n_right: int = np.sum(right_mask)
    
    left_entropy: float = calculate_entropy(labels[left_mask])
    right_entropy: float = calculate_entropy(labels[right_mask])
    
    weighted_entropy: float = (n_left / n_total) * left_entropy + (n_right / n_total) * right_entropy
    
    return parent_entropy - weighted_entropy
