import h5py
import os
import numpy as np

def preprocess_and_save(input_path, output_dir):
    # Open the .h5 file
    with h5py.File(input_path, 'r') as f:
        # Access the 'image' dataset
        images = f['image'][:]  # Replace 'image' with 'mask' if needed
        
        # (Optional) Normalize the data
        images = images / np.max(images)  # Normalize to [0, 1]

        # Save preprocessed images as .npy files
        for i, image in enumerate(images):
            output_path = os.path.join(output_dir, f"image_{i}.npy")
            np.save(output_path, image)
            print(f"Saved preprocessed image: {output_path}")

# Paths
input_h5 = 'D:/AI-Radiology-Assistant/data/raw/BraTS2020_training_data/volume_1_slice_0.h5'
output_dir = 'data/processed'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Preprocess and save
preprocess_and_save(input_h5, output_dir)
