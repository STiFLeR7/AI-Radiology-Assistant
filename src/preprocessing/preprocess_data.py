import os
import h5py
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

# Normalize data
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Augment data with rotation
def augment_data(data, angle=15):
    return rotate(data, angle=angle, reshape=False)

# Preprocess and save data
def preprocess_and_save(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    with h5py.File(input_path, 'r') as f:
        data = f['data'][:]
        normalized_data = normalize_data(data)
        augmented_data = augment_data(normalized_data)

        # Save normalized and augmented data
        np.save(os.path.join(output_path, 'normalized.npy'), normalized_data)
        np.save(os.path.join(output_path, 'augmented.npy'), augmented_data)

        print(f"Processed data saved in {output_path}")

# Example usage
if __name__ == "__main__":
    input_h5 = 'data/raw/BraTS2020_training_data/volume_1_slice_0.h5'
    output_dir = 'data/processed/sample/'
    preprocess_and_save(input_h5, output_dir)

    # Visualize processed data
    data = np.load(os.path.join(output_dir, 'augmented.npy'))
    plt.imshow(data, cmap='gray')
    plt.title('Augmented MRI Slice')
    plt.show()
