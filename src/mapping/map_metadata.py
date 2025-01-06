import os
import pandas as pd

def map_metadata(metadata_path, output_file):
    metadata = pd.read_csv(metadata_path)
    print(f"Processing {len(metadata)} entries")
    
    mapping = []
    for _, row in metadata.iterrows():
        if os.path.exists(row['slice_path']):
            mapping.append(row.to_dict())
    
    mapping_df = pd.DataFrame(mapping)
    mapping_df.to_csv(output_file, index=False)
    print(f"Saved {len(mapping_df)} valid entries to {output_file}")

if __name__ == "__main__":
    map_metadata("data/raw/updated_meta_data.csv", 
                "data/processed/mapped_metadata.csv")