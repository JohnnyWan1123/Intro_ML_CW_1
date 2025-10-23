#!/usr/bin/env python3
"""
Demo script to showcase the improved tree visualization using matplotlib.
This script builds a decision tree on a subset of the data and visualizes it.
"""

import numpy as np
from decision_tree_builder import decision_tree_learning

def load_dataset(filename='wifi_db/clean_dataset.txt'):
    """Load a dataset from file."""
    return np.loadtxt(filename)

if __name__ == "__main__":
    print("="*60)
    print("Decision Tree Visualization Demo")
    print("="*60)
    
    # Load clean dataset
    clean_dataset = load_dataset('wifi_db/clean_dataset.txt')
    print(f"\nLoaded clean dataset with shape: {clean_dataset.shape}")
    print(f"Using entire dataset for training")
    
    # Build decision tree
    print("\nBuilding decision tree...")
    tree = decision_tree_learning(clean_dataset, depth=0)
    
    print(f"✓ Tree built successfully!")
    print(f"  - Tree depth: {tree.get_depth()}")
    print(f"  - Number of leaves: {tree.get_leaf_count()}")
    
    # Visualize the tree
    print("\n" + "="*60)
    print("Generating matplotlib visualization...")
    print("="*60)
    print("Note: Large tree may take a moment to render...")
    
    # Show the tree visualization with larger canvas for the full tree
    tree.visualize_tree(figsize=(30, 20), save_path='tree_visualization_demo.png')
    
    print("\n✓ Visualization complete!")
    print("  - Image saved as: tree_visualization_demo.png")
    print("\nVisualization Features:")
    print("  • Blue boxes = Decision nodes (showing split conditions)")
    print("  • Green boxes = Leaf nodes (showing room predictions)")
    print("  • Edge labels show split direction (≤ or >)")
    print("  • Tree structure automatically laid out for clarity")

