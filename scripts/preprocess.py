import os
import shutil
import pandas as pd
from pathlib import Path
from PIL import Image

def preprocess_covid_chestxray(raw_dir, processed_dir, metadata_file):
    
    # Ensure processed directory exists
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_path = os.path.join(raw_dir, metadata_file)
    metadata = pd.read_csv(metadata_path)
    print(f"Loaded metadata with {len(metadata)} records.")

    # Filter metadata
    valid_views = ["PA", "AP"]
    metadata = metadata[metadata["view"].isin(valid_views)]
    metadata = metadata.dropna(subset=["finding", "filename"])  # Ensure no missing data
    print(f"Filtered metadata to {len(metadata)} records with valid views and findings.")

    # Process images
    processed_image_dir = os.path.join(processed_dir, "images")
    Path(processed_image_dir).mkdir(exist_ok=True)

    processed_metadata = []
    for _, row in metadata.iterrows():
        img_filename = row["filename"]
        finding = row["finding"]
        img_path = os.path.join(raw_dir, "images", img_filename)

        if not os.path.exists(img_path):
            print(f"Warning: Image file {img_filename} not found. Skipping...")
            continue

        try:
            # Validate and resize the image
            with Image.open(img_path) as img:
                img = img.convert("RGB")  # Ensure images are RGB
                img_resized = img.resize((224, 224))  # Resize for uniformity

            # Save processed image
            processed_img_path = os.path.join(processed_image_dir, img_filename)
            img_resized.save(processed_img_path)

            # Append metadata
            processed_metadata.append({
                "filename": img_filename,
                "finding": finding,
                "view": row["view"]
            })

        except Exception as e:
            print(f"Error processing image {img_filename}: {e}")

    # Save processed metadata
    processed_metadata_df = pd.DataFrame(processed_metadata)
    processed_metadata_file = os.path.join(processed_dir, "processed_metadata.csv")
    processed_metadata_df.to_csv(processed_metadata_file, index=False)
    print(f"Processed metadata saved to {processed_metadata_file}")

    print(f"Preprocessing complete. Processed {len(processed_metadata)} images.")

if __name__ == "__main__":
    raw_dir = "data/raw"  # Adjust the path to your raw dataset folder
    processed_dir = "data/processed"
    metadata_file = "metadata.csv"  # Adjust if the file name differs

    preprocess_covid_chestxray(raw_dir, processed_dir, metadata_file)
