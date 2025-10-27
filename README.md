# Intro to ML CW 1 - Decision Tree for WiFi Room Classification

A complete decision tree implementation using the ID3 algorithm to classify rooms based on WiFi signal strengths from multiple access points. This project includes tree building, visualization, cross-validation evaluation, and comprehensive performance metrics.

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
- **Cross-Validation Evaluation**: 10-fold cross-validation with comprehensive metrics:
  - Confusion matrices
  - Per-class precision, recall, and F1-scores
  - Overall accuracy with per-fold breakdown
  - Comparison between clean and noisy datasets
- **Automated Report Generation**: LaTeX report template with all evaluation results

## Quick Start

```bash
# Clone or navigate to the project directory
cd Intro_ML_CW_1

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the complete evaluation
python main.py
```

This will generate all visualizations and evaluation metrics in ~30 seconds.

## Installation

### 1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** If you encounter SSL/certificate issues, use:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

## Usage

### Complete Evaluation Pipeline (Recommended)

Run the main script to perform the full evaluation on both datasets:
```bash
python main.py
```

This comprehensive script will:
1. **Load Datasets**: Load both clean and noisy WiFi datasets (2000 samples each)
2. **Build Decision Trees**: Train trees on both datasets and display their structure
3. **Compare Trees**: Show tree statistics (depth, number of leaves) for both datasets
4. **Test Predictions**: Run sample predictions to verify tree functionality
5. **Generate Visualizations**: Create and save PNG visualizations:
   - `clean_tree_visualization.png` - Tree trained on clean data
   - `noisy_tree_visualization.png` - Tree trained on noisy data
   - `tree_visualization.png` - Copy for LaTeX report
6. **10-Fold Cross-Validation**: Perform comprehensive evaluation on both datasets:
   - Display confusion matrices (rows=actual, cols=predicted)
   - Show per-fold accuracy for each of the 10 folds
   - Calculate average accuracy across all folds
   - Report per-class precision, recall, and F1-scores for all 4 rooms

**Expected Output:**
```
Clean dataset shape: (2000, 8)
Noisy dataset shape: (2000, 8)

Building Decision Tree on Clean Dataset...
Tree depth: 14
Number of leaves: 44

10-FOLD CROSS-VALIDATION
Dataset: Clean
Average accuracy: 0.974 (97.4%)
Room 1: Precision=0.986, Recall=0.988, F1=0.987
...

Dataset: Noisy
Average accuracy: 0.804 (80.4%)
...
```

### Visualization Demo

For a focused demonstration of the visualization feature only:
```bash
python visualize_demo.py
```

This creates a tree trained on the entire clean dataset and generates a detailed visualization saved as `tree_visualization_demo.png`.

### Using in Your Code

#### Building and Using a Decision Tree

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

#### Running Cross-Validation Evaluation

```python
from evaluation import k_fold_cross_validation, evaluate
import numpy as np

# Load dataset
dataset = np.loadtxt('wifi_db/clean_dataset.txt')

# Run 10-fold cross-validation
results = k_fold_cross_validation(dataset, k=10, random_seed=42)

# Access results
print(f"Average accuracy: {results['average_accuracy']:.3f}")
print(f"Confusion matrix:\n{results['confusion_matrix']}")
print(f"Per-fold accuracies: {results['accuracy_per_fold']}")

# Per-class metrics for each room
for label, precision, recall, f1 in zip(
    results['labels'],
    results['precision'], 
    results['recall'],
    results['f1']
):
    print(f"Room {label}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
```

#### Evaluating a Single Train/Test Split

```python
from evaluation import evaluate
from decision_tree_builder import decision_tree_learning
import numpy as np

# Split your data
train_data = dataset[:1600]  # 80% for training
test_data = dataset[1600:]    # 20% for testing

# Train tree on training set
tree = decision_tree_learning(train_data, depth=0)

# Evaluate on test set
accuracy = evaluate(test_data, tree)
print(f"Test accuracy: {accuracy:.3f}")
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
├── decision_tree_builder.py      # Core ID3 algorithm implementation
├── tree_node.py                  # TreeNode class with visualization & prediction
├── evaluation.py                 # Cross-validation and evaluation metrics
├── main.py                       # Complete evaluation pipeline with CV
├── visualize_demo.py             # Visualization-focused demo
├── requirements.txt              # Python dependencies
├── report.tex                    # LaTeX report template with results
├── wifi_db/
│   ├── clean_dataset.txt        # Clean WiFi signal dataset (2000 samples)
│   └── noisy_dataset.txt        # Noisy WiFi signal dataset (2000 samples)
└── Generated Output Files:
    ├── clean_tree_visualization.png       # Tree visualization for clean data
    ├── noisy_tree_visualization.png       # Tree visualization for noisy data
    └── tree_visualization.png             # Copy for LaTeX report
```

## Dataset Format

The WiFi dataset contains:
- **7 feature columns**: Signal strengths from WiFi access points (AP1-AP7)
  - Values are continuous, typically in dBm (negative values)
  - Range approximately from -90 to -30 dBm
- **1 label column**: Room number classification (1, 2, 3, or 4)
- **2000 samples total**: 500 samples per room class

**Dataset Files:**
- `clean_dataset.txt`: Original clean WiFi signal measurements
- `noisy_dataset.txt`: Same data with added noise for robustness testing

## Output Files

After running `main.py`, the following files will be generated:

### Visualizations
- **`clean_tree_visualization.png`**: Complete decision tree trained on clean dataset
  - Shows tree depth (~14 levels) and structure (~44 leaf nodes)
- **`noisy_tree_visualization.png`**: Complete decision tree trained on noisy dataset
  - Typically deeper (~18 levels) with more leaves (~335 nodes) due to overfitting
- **`tree_visualization.png`**: Copy of clean tree for LaTeX report inclusion

### Report
- **`report.tex`**: LaTeX report with all evaluation metrics pre-filled
  - Compile with: `pdflatex report.tex`
  - Contains confusion matrices, accuracy metrics, and analysis

## Evaluation Metrics Explained

The evaluation module (`evaluation.py`) provides comprehensive metrics:

- **Confusion Matrix**: 4×4 matrix showing actual vs predicted room classifications
- **Accuracy**: Overall classification accuracy (correct predictions / total predictions)
- **Precision**: Per-class metric - of all predicted Room X, how many were actually Room X
- **Recall**: Per-class metric - of all actual Room X samples, how many were correctly predicted
- **F1-Score**: Harmonic mean of precision and recall (2 × P × R / (P + R))

### Expected Results Summary

| Metric | Clean Dataset | Noisy Dataset |
|--------|---------------|---------------|
| Overall Accuracy | ~97.4% | ~80.4% |
| Tree Depth | ~14 | ~18 |
| Number of Leaves | ~44 | ~335 |
| Best Performing Room | Room 4 (F1: 0.991) | Room 3 (F1: 0.810) |
| Most Confused Rooms | Rooms 2 & 3 | Rooms 2 & 3 |

## Requirements

- **Python 3.12+** (tested on Python 3.12)
- **numpy 2.1.1** - Array operations and numerical computing
- **matplotlib 3.9.2** - Tree visualization and plotting

**Note:** This project only uses Numpy, Matplotlib, and standard Python libraries (typing, dataclasses) as required by the coursework specification. No external libraries like scikit-learn or scipy are used.

All dependencies are listed in `requirements.txt` for easy installation.

## Algorithm Details

### ID3 Decision Tree Implementation

The implementation follows the classic ID3 algorithm:

1. **Information Gain**: Uses entropy-based information gain to select the best split at each node
2. **Splitting Strategy**: Binary splits on continuous features (threshold-based)
3. **Stopping Criteria**: 
   - All samples belong to the same class (pure leaf)
   - No features remain to split on
   - Dataset becomes empty
4. **Greedy Approach**: Makes locally optimal decisions at each node

### Cross-Validation

- **K-Fold Strategy**: Dataset randomly shuffled and split into k equal folds
- **Training/Testing**: Each fold serves as test set once, remaining k-1 folds for training
- **Aggregation**: Metrics aggregated across all folds for robust performance estimation
- **Reproducibility**: Fixed random seed (42) ensures consistent results

## Troubleshooting

### Common Issues

**Problem: `python: command not found`**
```bash
# Try using python3 instead
python3 main.py
```

**Problem: `ModuleNotFoundError: No module named 'numpy'`**
```bash
# Make sure virtual environment is activated
source venv/bin/activate
# Reinstall dependencies
pip install -r requirements.txt
```

**Problem: Visualization doesn't display**
```bash
# The visualizations are saved as PNG files automatically
# Check the directory for:
# - clean_tree_visualization.png
# - noisy_tree_visualization.png
```

**Problem: Out of memory when visualizing large trees**
```python
# Reduce figure size in visualize_tree() call
tree.visualize_tree(figsize=(15, 10))  # Instead of (30, 20)
```

**Problem: Different results than expected**
```python
# Ensure using the correct random seed for reproducibility
results = k_fold_cross_validation(dataset, k=10, random_seed=42)
```

## File Descriptions

### Core Implementation
- **`decision_tree_builder.py`**: Contains `decision_tree_learning()` function implementing ID3
  - Entropy calculation
  - Information gain computation
  - Recursive tree building
  
- **`tree_node.py`**: TreeNode class with:
  - Tree structure (left/right children, split conditions)
  - Prediction methods (single sample and batch)
  - Visualization using matplotlib
  - Utility methods (depth, leaf count)

- **`evaluation.py`**: Evaluation utilities:
  - `evaluate()`: Calculate accuracy on test set
  - `k_fold_cross_validation()`: Perform k-fold CV
  - Internal metrics computation (confusion matrix, precision, recall, F1)

### Scripts
- **`main.py`**: Complete pipeline demonstrating all functionality
- **`visualize_demo.py`**: Focused visualization demonstration

### Data
- **`wifi_db/clean_dataset.txt`**: Original WiFi measurements (2000×8 array)
- **`wifi_db/noisy_dataset.txt`**: Noisy version for robustness testing

### Documentation
- **`report.tex`**: LaTeX report template with all results
- **`README.md`**: This file
- **`requirements.txt`**: Python package dependencies

## Performance Notes

- **Training time**: ~1-2 seconds per tree on full dataset
- **Cross-validation time**: ~20-30 seconds per dataset (10 folds)
- **Visualization generation**: ~5-10 seconds depending on tree size
- **Memory usage**: ~50-100 MB for typical operation

The noisy dataset produces significantly more complex trees (335 vs 44 leaves) due to overfitting to noise, resulting in longer visualization times.

## License

This is a coursework project for COMP60012: Introduction to Machine Learning at Imperial College London.