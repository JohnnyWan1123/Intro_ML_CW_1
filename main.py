import numpy as np

def load_datasets():
    """
    Load the clean and noisy WiFi datasets.
    
    Returns:
        tuple: (clean_data, noisy_data) where each dataset is a 2000x8 array
               - First 7 columns: WiFi signal strengths (continuous features)
               - Last column: Room number (label)
    """
    # Load clean dataset
    clean_data = np.loadtxt('wifi_db/clean_dataset.txt')
    
    # Load noisy dataset  
    noisy_data = np.loadtxt('wifi_db/noisy_dataset.txt')
    
    print(f"Clean dataset shape: {clean_data.shape}")
    print(f"Noisy dataset shape: {noisy_data.shape}")
    print(f"Clean dataset - First 5 samples:")
    print(clean_data[:5])
    print(f"Noisy dataset - First 5 samples:")
    print(noisy_data[:5])
    
    return clean_data, noisy_data

if __name__ == "__main__":
    # Load the datasets
    clean_dataset, noisy_dataset = load_datasets()
    
    # Extract features and labels
    clean_features = clean_dataset[:, :-1]  # First 7 columns (WiFi signals)
    clean_labels = clean_dataset[:, -1]    # Last column (room numbers)
    
    noisy_features = noisy_dataset[:, :-1]  # First 7 columns (WiFi signals)
    noisy_labels = noisy_dataset[:, -1]    # Last column (room numbers)
    
    print(f"\nClean dataset - Features shape: {clean_features.shape}, Labels shape: {clean_labels.shape}")
    print(f"Noisy dataset - Features shape: {noisy_features.shape}, Labels shape: {noisy_labels.shape}")
    
    # Check unique room numbers
    print(f"Unique room numbers in clean dataset: {np.unique(clean_labels)}")
    print(f"Unique room numbers in noisy dataset: {np.unique(noisy_labels)}")
