# Intro to ML CW 1 - Decision Tree for WiFi Room Classification

A decision tree implementation using the ID3 algorithm to classify rooms based on WiFi signal strengths from multiple access points.

## Features

- **Decision Tree Learning**: ID3 algorithm implementation with information gain-based splitting
- **Tree Visualization**: Beautiful matplotlib-based tree visualization with:
  - Color-coded nodes (blue for decision nodes, green for leaf nodes)
  - Split conditions clearly labeled
  - Edge labels showing split direction (≤ or >)
  - Automatic layout optimization
  - High-resolution PNG export
- **Prediction**: Support for both single and batch predictions
- **Tree Metrics**: Calculate tree depth and leaf count

## Installation

### 1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

## Usage

### Basic Usage

Run the main script to build trees on both clean and noisy datasets:
```bash
python main.py
```

This will:
- Build decision trees on clean and noisy datasets
- Display tree statistics (depth, number of leaves)
- Run predictions and calculate accuracy
- Generate and display matplotlib visualizations

### Visualization Demo

For a focused demonstration of the visualization feature:
```bash
python visualize_demo.py
```

This creates a tree trained from the entre clean dataset and displays a detailed visualization.

### Using in Your Code

```python
from decision_tree_builder import decision_tree_learning
import numpy as np

# Load your dataset (features + labels in last column)
dataset = np.loadtxt('wifi_db/clean_dataset.txt')

# Build the decision tree
tree = decision_tree_learning(dataset, depth=0)

# Visualize the tree
tree.visualize_tree(figsize=(20, 12), save_path='my_tree.png')

# Make predictions
prediction = tree.predict(sample_features)
predictions = tree.predict_batch(multiple_samples)

# Get tree metrics
depth = tree.get_depth()
num_leaves = tree.get_leaf_count()
```

## Visualization Features

The `visualize_tree()` method provides a comprehensive visualization:

- **Decision Nodes (Blue boxes)**: Show which WiFi access point and threshold value to split on
- **Leaf Nodes (Green boxes)**: Show the predicted room number
- **Edge Labels**: 
  - `≤` indicates values less than or equal to threshold (left branch)
  - `>` indicates values greater than threshold (right branch)
- **Legend**: Clearly identifies node types
- **Title**: Shows tree depth and number of leaves

### Customization Options

```python
tree.visualize_tree(
    figsize=(20, 12),              # Figure size (width, height)
    save_path='visualization.png'  # Optional: save to file
)
```

## Project Structure

```
.
├── decision_tree_builder.py  # Core ID3 algorithm implementation
├── tree_node.py              # TreeNode class with visualization
├── main.py                   # Main script demonstrating all features
├── visualize_demo.py         # Visualization-focused demo
├── requirements.txt          # Python dependencies
└── wifi_db/
    ├── clean_dataset.txt     # Clean training data
    └── noisy_dataset.txt     # Noisy training data
```

## Dataset Format

The WiFi dataset contains:
- 7 feature columns: Signal strengths from WiFi access points
- 1 label column: Room number (1-4)
- 2000 samples total

## Requirements

- Python 3.12+
- numpy 2.1.1
- matplotlib 3.9.2
- scipy 1.14.1

## License

This is a coursework project for Introduction to Machine Learning.