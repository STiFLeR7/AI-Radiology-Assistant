import os
import pandas as pd

def update_metadata(metadata_path, output_path):
    """
    Updates the metadata file by removing entries with missing slice paths.

    Args:
        metadata_path (str): Path to the original metadata CSV file.
        output_path (str): Path to save the updated metadata CSV file.
    """
    # Load the metadata CSV
    metadata = pd.read_csv(metadata_path)

    # Filter rows where the slice path exists
    valid_metadata = metadata[metadata['slice_path'].apply(os.path.exists)]

    # Save the updated metadata
    valid_metadata.to_csv(output_path, index=False)
    print(f"Updated metadata saved to: {output_path}")

# Main script execution
if __name__ == "__main__":
    metadata_file = "data/raw/meta_data.csv"  # Path to the original metadata file
    updated_metadata_file = "data/raw/updated_meta_data.csv"  # Path to save the updated metadata

    update_metadata(metadata_file, updated_metadata_file)
