import h5py
import matplotlib.pyplot as plt

# Load and explore .h5 file
def explore_h5(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f"Keys in {file_path}: {list(f.keys())}")
        
        # Explore and visualize the 'image' data
        if 'image' in f:
            image_data = f['image'][:]  # Access the 'image' key
            print(f"Image data shape: {image_data.shape}")
            
            # Display the first slice of the image data
            plt.imshow(image_data[:, :, 0], cmap='gray')  # Adjust slicing as needed
            plt.title(f"Sample MRI Image Slice from {file_path}")
            plt.colorbar()
            plt.show()
        
        # Explore and visualize the 'mask' data
        if 'mask' in f:
            mask_data = f['mask'][:]  # Access the 'mask' key
            print(f"Mask data shape: {mask_data.shape}")
            
            # Display the first slice of the mask data
            plt.imshow(mask_data[:, :, 0], cmap='viridis')  # Adjust slicing as needed
            plt.title(f"Sample Mask Slice from {file_path}")
            plt.colorbar()
            plt.show()

# Example usage
if __name__ == "__main__":
    h5_file_path = 'data/raw/BraTS2020_training_data/volume_1_slice_0.h5'
    explore_h5(h5_file_path)
