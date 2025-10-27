import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional, Dict, Tuple
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
    
    def visualize_tree(self, figsize: Tuple[int, int] = None, save_path: Optional[str] = None) -> None:
        """
        Visualize the decision tree using matplotlib with a beautiful layout.
        
        Args:
            figsize: Size of the figure (width, height). If None, automatically calculated based on tree size.
            save_path: If provided, save the figure to this path
        """
        # Automatically calculate figure size based on tree dimensions if not provided
        if figsize is None:
            max_depth = self.get_depth()
            leaf_count = self.get_leaf_count()
            # Width scales with number of leaves, height with depth
            width = max(16, min(40, leaf_count * 2))
            height = max(10, min(30, max_depth * 3))
            figsize = (width, height)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Calculate positions for all nodes
        positions = self._calculate_positions()
        
        # Draw edges first (so they appear behind nodes)
        self._draw_edges(ax, positions)
        
        # Draw nodes
        self._draw_nodes(ax, positions)
        
        # Add title
        max_depth = self.get_depth()
        leaf_count = self.get_leaf_count()
        plt.title(f'Decision Tree Visualization\nDepth: {max_depth} | Leaves: {leaf_count}', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='#90EE90', edgecolor='#2E7D32', label='Leaf Node (Room)'),
            mpatches.Patch(facecolor='#87CEEB', edgecolor='#1976D2', label='Decision Node (Split)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Tree visualization saved to: {save_path}")
        
        plt.show()
    
    def _calculate_positions(self) -> Dict[int, Tuple[float, float]]:
        """
        Calculate (x, y) positions for all nodes in the tree.
        
        Returns:
            Dictionary mapping node id to (x, y) position
        """
        positions = {}
        max_depth = self.get_depth()
        
        # Assign unique IDs to nodes and calculate their positions
        self._assign_positions_recursive(positions, 0, 1, 0, max_depth)
        
        return positions
    
    def _assign_positions_recursive(self, positions: Dict[int, Tuple[float, float]], 
                                   left: float, right: float, depth: int, 
                                   max_depth: int) -> int:
        """
        Recursively assign positions to nodes.
        
        Args:
            positions: Dictionary to store positions
            left: Left boundary of x-coordinate
            right: Right boundary of x-coordinate
            depth: Current depth
            max_depth: Maximum depth of tree
            
        Returns:
            Node ID
        """
        node_id = id(self)
        
        # Calculate y position (inverted so root is at top)
        y = 1 - (depth / (max_depth + 1))
        
        if self.is_leaf:
            # Leaf node - center in the allocated space
            x = (left + right) / 2
            positions[node_id] = (x, y)
        else:
            # Internal node - position based on children
            # Allocate space proportionally based on number of leaves in each subtree
            left_leaves = self.left_child.get_leaf_count()
            right_leaves = self.right_child.get_leaf_count()
            total_leaves = left_leaves + right_leaves
            
            # Calculate split point based on leaf proportions
            available_width = right - left
            mid = left + (available_width * left_leaves / total_leaves)
            
            # Recursively position children
            left_id = self.left_child._assign_positions_recursive(
                positions, left, mid, depth + 1, max_depth
            )
            right_id = self.right_child._assign_positions_recursive(
                positions, mid, right, depth + 1, max_depth
            )
            
            # Position this node between its children
            left_pos = positions[left_id]
            right_pos = positions[right_id]
            x = (left_pos[0] + right_pos[0]) / 2
            positions[node_id] = (x, y)
        
        return node_id
    
    def _draw_edges(self, ax, positions: Dict[int, Tuple[float, float]]) -> None:
        """Draw edges between parent and child nodes."""
        self._draw_edges_recursive(ax, positions)
    
    def _draw_edges_recursive(self, ax, positions: Dict[int, Tuple[float, float]]) -> None:
        """Recursively draw edges."""
        if not self.is_leaf:
            node_id = id(self)
            parent_pos = positions[node_id]
            
            # Draw edge to left child
            left_id = id(self.left_child)
            left_pos = positions[left_id]
            ax.plot([parent_pos[0], left_pos[0]], [parent_pos[1], left_pos[1]], 
                   'k-', linewidth=2, alpha=0.6, zorder=1)
            
            # Add label for left edge (<=)
            mid_x = (parent_pos[0] + left_pos[0]) / 2
            mid_y = (parent_pos[1] + left_pos[1]) / 2
            ax.text(mid_x - 0.01, mid_y, '≤', fontsize=9, color='#1976D2',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='#1976D2', alpha=0.8))
            
            # Draw edge to right child
            right_id = id(self.right_child)
            right_pos = positions[right_id]
            ax.plot([parent_pos[0], right_pos[0]], [parent_pos[1], right_pos[1]], 
                   'k-', linewidth=2, alpha=0.6, zorder=1)
            
            # Add label for right edge (>)
            mid_x = (parent_pos[0] + right_pos[0]) / 2
            mid_y = (parent_pos[1] + right_pos[1]) / 2
            ax.text(mid_x + 0.01, mid_y, '>', fontsize=9, color='#D32F2F',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='#D32F2F', alpha=0.8))
            
            # Recursively draw edges for children
            self.left_child._draw_edges_recursive(ax, positions)
            self.right_child._draw_edges_recursive(ax, positions)
    
    def _draw_nodes(self, ax, positions: Dict[int, Tuple[float, float]]) -> None:
        """Draw all nodes in the tree."""
        self._draw_nodes_recursive(ax, positions)
    
    def _draw_nodes_recursive(self, ax, positions: Dict[int, Tuple[float, float]]) -> None:
        """Recursively draw nodes."""
        node_id = id(self)
        x, y = positions[node_id]
        
        if self.is_leaf:
            # Leaf node - green box with room number
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='#90EE90', 
                            edgecolor='#2E7D32', linewidth=2.5)
            ax.text(x, y, f'Room {int(self.label)}', 
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   bbox=bbox_props, zorder=3)
        else:
            # Decision node - blue box with split condition
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='#87CEEB', 
                            edgecolor='#1976D2', linewidth=2.5)
            
            split_text = f'WiFi AP{self.feature_index + 1}\n≤ {self.threshold:.2f}'
            ax.text(x, y, split_text, 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=bbox_props, zorder=3)
            
            # Recursively draw children
            self.left_child._draw_nodes_recursive(ax, positions)
            self.right_child._draw_nodes_recursive(ax, positions)
