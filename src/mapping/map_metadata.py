import pandas as pd
import os

# Map metadata to .h5 files
def map_metadata(metadata_path, h5_dir):
    metadata = pd.read_excel(metadata_path)
    h5_files = os.listdir(h5_dir)

    # Assuming patient IDs are part of the file names
    metadata['mapped_files'] = metadata['PatientID'].apply(
        lambda x: [f for f in h5_files if str(x) in f]
    )

    print("Metadata mapping complete:")
    print(metadata.head())
    return metadata

# Save mapping
def save_mapping(metadata, output_path):
    metadata.to_csv(output_path, index=False)
    print(f"Mapping saved to {output_path}")

# Example usage
if __name__ == "__main__":
    metadata_file = 'data/raw/meta_data.xlsx'
    h5_folder = 'data/raw/BraTS2020_training_data/'
    output_csv = 'data/processed/metadata_mapping.csv'

    metadata = map_metadata(metadata_file, h5_folder)
    save_mapping(metadata, output_csv)
