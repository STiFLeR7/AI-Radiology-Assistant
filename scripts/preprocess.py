import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

input_dir = "D:/AI-Radiology-Assistant/data/raw/images/"
output_dir = "D:/AI-Radiology-Assistant/data/preprocessed_images/"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".nii.gz"):
        file_path = os.path.join(input_dir, filename)
        # Load the NIfTI file
        nifti = nib.load(file_path)
        data = nifti.get_fdata()
        # Take the middle slice along the third dimension
        slice_idx = data.shape[2] // 2
        slice_data = data[:, :, slice_idx]
        # Normalize the slice
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
        # Save the slice as a PNG
        output_path = os.path.join(output_dir, filename.replace(".nii.gz", ".png"))
        plt.imsave(output_path, slice_data, cmap="gray")
