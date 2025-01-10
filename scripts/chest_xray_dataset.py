import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, image_dir, metadata, transform=None):
        self.image_dir = image_dir
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Access the image filename
        img_name = os.path.join(self.image_dir, self.metadata.iloc[idx]['filename'])

        # Load the image
        image = Image.open(img_name).convert('RGB')

        # Access the label (use 'encoded_label' instead of 'label')
        label = self.metadata.iloc[idx]['encoded_label']

        # Apply any transformations
        if self.transform:
            image = self.transform(image)

        return image, label
