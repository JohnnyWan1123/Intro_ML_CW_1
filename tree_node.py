import numpy as np
from typing import Optional
from dataclasses import dataclass

@dataclass
class TreeNode:
    """
    A node in a decision tree that can be either a leaf or an internal node.
    
    Attributes:
        is_leaf (bool): True if this is a leaf node, False if internal node
        label (Optional[int]): Class label (only for leaf nodes)
        feature_index (Optional[int]): Feature index to split on (only for internal nodes)
        threshold (Optional[float]): Threshold value for splitting (only for internal nodes)
        left_child (Optional['TreeNode']): Left subtree (values <= threshold)
        right_child (Optional['TreeNode']): Right subtree (values > threshold)
        depth (int): Depth of this node in the tree
    """
    is_leaf: bool
    label: Optional[int] = None
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left_child: Optional['TreeNode'] = None
    right_child: Optional['TreeNode'] = None
    depth: int = 0
    
    def print_tree(self, indent: int = 0) -> None:
        """Print the tree structure for visualization."""
        if self.is_leaf:
            print("  " * indent + f"Leaf: Room {self.label} (depth: {self.depth})")
        else:
            print("  " * indent + f"Split on WiFi AP{self.feature_index + 1} <= {self.threshold:.2f} (depth: {self.depth})")
            print("  " * indent + "├─ Left (≤):")
            self.left_child.print_tree(indent + 1)
            print("  " * indent + "└─ Right (>):")
            self.right_child.print_tree(indent + 1)
    
    def get_depth(self) -> int:
        """Get the maximum depth of the tree."""
        if self.is_leaf:
            return self.depth
        return max(self.left_child.get_depth(), self.right_child.get_depth())
    
    def get_leaf_count(self) -> int:
        """Get the total number of leaf nodes."""
        if self.is_leaf:
            return 1
        return self.left_child.get_leaf_count() + self.right_child.get_leaf_count()
    
    def predict(self, sample: np.ndarray) -> int:
        """Predict the class label for a single sample."""
        if self.is_leaf:
            return self.label
        
        if sample[self.feature_index] <= self.threshold:
            return self.left_child.predict(sample)
        else:
            return self.right_child.predict(sample)
    
    def predict_batch(self, samples: np.ndarray) -> np.ndarray:
        """Predict class labels for multiple samples."""
        predictions = np.zeros(samples.shape[0], dtype=int)
        for i, sample in enumerate(samples):
            predictions[i] = self.predict(sample)
        return predictions
