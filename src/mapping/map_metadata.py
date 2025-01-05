import os
import pandas as pd

def map_metadata(metadata_path, h5_folder, output_file):
    """
    Maps metadata to corresponding H5 slices and saves the mapping to a CSV file.

    Args:
        metadata_path (str): Path to the metadata CSV file.
        h5_folder (str): Path to the folder containing H5 files.
        output_file (str): Path to save the mapping CSV.
    """
    # Load the metadata CSV
    metadata = pd.read_csv(metadata_path)

    # Prepare a list to hold the mapping
    mapping = []

    # Iterate through the rows of the metadata
    for _, row in metadata.iterrows():
        subject_id = row['BraTS_2020_subject_ID']  # Adjusted to match your column
        slices = row['HGG/LGG']  # Adjusted to match your column
        target = row['MGMT_value']  # Adjusted to match your column

        # Construct the H5 file path
        h5_file = os.path.join(h5_folder, f"{subject_id}.h5")

        # Check if the file exists
        if os.path.exists(h5_file):
            mapping.append({'file_path': h5_file, 'target': target})
        else:
            print(f"Warning: H5 file for {subject_id} not found in {h5_folder}")

    # Convert the mapping to a DataFrame and save as CSV
    mapping_df = pd.DataFrame(mapping)
    mapping_df.to_csv(output_file, index=False)
    print(f"Mapping saved to: {output_file}")

# Main script execution
if __name__ == "__main__":
    metadata_file = "data/raw/meta_data.csv"  # Path to the metadata file
    h5_folder = "data/raw/BraTS2020_training_data"  # Path to the folder containing H5 files
    output_file = "data/processed/mapped_metadata.csv"  # Path to save the mapping CSV

    map_metadata(metadata_file, h5_folder, output_file)
