import numpy as np
from typing import Tuple
from decision_tree_builder import decision_tree_learning
from evaluation import k_fold_cross_validation

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

if __name__ == "__main__":
    # Load datasets
    clean_dataset, noisy_dataset = load_datasets()
    print(f"Loaded datasets: {clean_dataset.shape[0]} samples each\n")
    
    # Build decision trees
    print("="*60)
    print("BUILDING DECISION TREES")
    print("="*60)
    tree = decision_tree_learning(clean_dataset, depth=0)
    noisy_tree = decision_tree_learning(noisy_dataset, depth=0)
    
    print(f"Clean tree  - Depth: {tree.get_depth():2d} | Leaves: {tree.get_leaf_count():3d}")
    print(f"Noisy tree  - Depth: {noisy_tree.get_depth():2d} | Leaves: {noisy_tree.get_leaf_count():3d}")
    
    # Save visualizations
    print(f"\n" + "="*60)
    print("SAVING VISUALIZATIONS")
    print("="*60)
    tree.visualize_tree(save_path='clean_tree_visualization.png', show=False)
    print("✓ clean_tree_visualization.png")
    noisy_tree.visualize_tree(save_path='noisy_tree_visualization.png', show=False)
    print("✓ noisy_tree_visualization.png")

    # Run 10-fold cross-validation
    print(f"\n" + "="*60)
    print("10-FOLD CROSS-VALIDATION")
    print("="*60)

    for dataset_name, dataset in (("Clean", clean_dataset), ("Noisy", noisy_dataset)):
        print(f"\n{dataset_name} Dataset:")
        cv_results = k_fold_cross_validation(dataset, k=10, random_seed=42)

        print(f"Average accuracy: {cv_results['average_accuracy']:.3f}")
        print(f"Per-fold: {' '.join(f'{acc:.3f}' for acc in cv_results['accuracy_per_fold'])}")
        
        print("\nConfusion matrix (rows=actual, cols=predicted):")
        print(cv_results["confusion_matrix"])

        print("\nPer-class metrics:")
        for label, precision, recall, f1_score in zip(
            cv_results["labels"],
            cv_results["precision"],
            cv_results["recall"],
            cv_results["f1"],
        ):
            print(f"  Room {label}: P={precision:.3f} R={recall:.3f} F1={f1_score:.3f}")
    
    print(f"\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)