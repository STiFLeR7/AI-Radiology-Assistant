import os
from torch.utils.data import Dataset
from PIL import Image

class ChestXrayDataset(Dataset):
    def __init__(self, image_dir, metadata, transform=None):
        self.image_dir = image_dir
        self.metadata = metadata
        self.transform = transform
        # Filter metadata to only include supported image files
        self.metadata = self.metadata[self.metadata['filename'].str.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_name).convert('RGB')  # Open image and convert to RGB
        if self.transform:
            image = self.transform(image)
        label = row['encoded_label']
        return image, label
