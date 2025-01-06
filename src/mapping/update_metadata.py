import os
import pandas as pd

def fix_path(path):
    """Convert Colab path to local path"""
    return os.path.join('data', path.split('/data/')[-1])

def update_metadata(metadata_path, output_path):
    # Read metadata
    metadata = pd.read_csv(metadata_path)
    
    # Fix paths
    metadata['slice_path'] = metadata['slice_path'].apply(fix_path)
    
    # Save all entries (we'll check existence in map_metadata)
    metadata.to_csv(output_path, index=False)
    print(f"Saved {len(metadata)} entries")

if __name__ == "__main__":
    update_metadata("data/raw/meta_data.csv", "data/raw/updated_meta_data.csv")