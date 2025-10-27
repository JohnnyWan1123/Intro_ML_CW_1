# Decision Tree for WiFi Room Classification

A complete decision tree implementation using the ID3 algorithm to classify rooms based on WiFi signal strengths. Includes visualization, cross-validation, and comprehensive evaluation metrics.

## Features

- **ID3 Decision Tree**: Information gain-based splitting with entropy calculation
- **Smart Visualization**: 
  - Color-coded nodes (blue=decision, green=leaf)
  - Automatic layout preventing overlap
  - Auto-sizing based on tree dimensions
  - High-resolution PNG export
- **Evaluation**: 10-fold cross-validation with confusion matrix, precision, recall, and F1-scores
- **Prediction**: Single sample and batch prediction support

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Run complete evaluation
python main.py
```

This runs the full pipeline (tree building, visualization, cross-validation) in ~30 seconds without user interaction.

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.12+, NumPy 2.1.1, Matplotlib 3.9.2

## Usage

### Complete Evaluation (Recommended)

```bash
python main.py
```

**What it does:**
1. Builds decision trees on clean and noisy datasets
2. Saves visualizations to `clean_tree_visualization.png` and `noisy_tree_visualization.png`
3. Runs 10-fold cross-validation on both datasets
4. Displays confusion matrices and per-class metrics

**Expected Results:**
- Clean dataset: ~97.4% accuracy, depth ~14, ~44 leaves
- Noisy dataset: ~80.4% accuracy, depth ~18, ~335 leaves

### Visualization Only

```bash
python visualize_demo.py
```

Creates `tree_visualization.png` with a tree trained on the clean dataset.

### Using in Your Code

#### Build and Visualize a Tree

```python
from decision_tree_builder import decision_tree_learning
import numpy as np

# Load dataset (features + label in last column)
dataset = np.loadtxt('wifi_db/clean_dataset.txt')

# Build tree
tree = decision_tree_learning(dataset, depth=0)

# Visualize (automatic sizing, non-blocking)
tree.visualize_tree(save_path='my_tree.png', show=False)

# Or with custom size and display
tree.visualize_tree(figsize=(20, 12), save_path='my_tree.png', show=True)

# Make predictions
prediction = tree.predict(sample_features)
batch_predictions = tree.predict_batch(multiple_samples)

# Get metrics
print(f"Depth: {tree.get_depth()}, Leaves: {tree.get_leaf_count()}")
```

#### Run Cross-Validation

```python
from evaluation import k_fold_cross_validation

# Run 10-fold CV
results = k_fold_cross_validation(dataset, k=10, random_seed=42)

# Access results
print(f"Average accuracy: {results['average_accuracy']:.3f}")
print(f"Confusion matrix:\n{results['confusion_matrix']}")

# Per-class metrics
for label, precision, recall, f1 in zip(
    results['labels'], results['precision'], 
    results['recall'], results['f1']
):
    print(f"Room {label}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
```

#### Single Train/Test Split

```python
from evaluation import evaluate

# Split data
train_data = dataset[:1600]
test_data = dataset[1600:]

# Train and evaluate
tree = decision_tree_learning(train_data, depth=0)
accuracy = evaluate(test_data, tree)
print(f"Accuracy: {accuracy:.3f}")
```

## Visualization Options

### Parameters

```python
tree.visualize_tree(
    figsize=None,        # (width, height) or None for auto-sizing
    save_path=None,      # Path to save PNG, or None for display only
    show=True            # True to display, False to save without blocking
)
```

### Examples

```python
# Automatic sizing, save only (recommended for automation)
tree.visualize_tree(save_path='tree.png', show=False)

# Custom size with display
tree.visualize_tree(figsize=(20, 12), save_path='tree.png', show=True)

# Display only
tree.visualize_tree(show=True)
```

## Project Structure

```
.
├── decision_tree_builder.py    # ID3 algorithm implementation
├── tree_node.py                # TreeNode class with visualization
├── evaluation.py               # Cross-validation and metrics
├── main.py                     # Complete evaluation pipeline
├── visualize_demo.py           # Visualization demo
├── requirements.txt            # Dependencies
├── wifi_db/
│   ├── clean_dataset.txt      # Clean WiFi signals (2000×8)
│   └── noisy_dataset.txt      # Noisy WiFi signals (2000×8)
└── Output:
    ├── clean_tree_visualization.png
    └── noisy_tree_visualization.png
```

## Dataset Format

- **7 feature columns**: WiFi signal strengths (dBm) from AP1-AP7
- **1 label column**: Room number (1, 2, 3, or 4)
- **2000 samples**: 500 per room class

## Troubleshooting

**Virtual environment not activated:**
```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

**Module not found:**
```bash
pip install -r requirements.txt
```

**Visualization doesn't display:**
- By default, `main.py` saves without displaying (`show=False`)
- To display: `tree.visualize_tree(save_path='tree.png', show=True)`

**Nodes overlapping:**
- Use automatic sizing: `tree.visualize_tree(save_path='tree.png')`
- Or increase size: `tree.visualize_tree(figsize=(30, 20), save_path='tree.png')`

## Algorithm Details

**ID3 Decision Tree:**
- Entropy-based information gain for split selection
- Binary splits on continuous features (threshold-based)
- Stops when: all samples same class, no features remain, or dataset empty

**10-Fold Cross-Validation:**
- Dataset shuffled and split into 10 equal folds
- Each fold used once as test set
- Fixed random seed (42) for reproducibility

## Performance

- **Training**: ~1-2 seconds per tree
- **Cross-validation**: ~20-30 seconds per dataset
- **Visualization**: ~5-10 seconds
- **Memory**: ~50-100 MB

## License

Coursework project for COMP60012: Introduction to Machine Learning at Imperial College London.
