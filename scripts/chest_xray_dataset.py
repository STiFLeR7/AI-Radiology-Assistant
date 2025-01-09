import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ChestXrayDataset(Dataset):
    def __init__(self, metadata, img_dir, transform=None):
        """
        Args:
            metadata (pd.DataFrame): Metadata DataFrame containing 'filename' and 'label' columns.
            img_dir (str): Directory where images are stored.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.metadata = metadata
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            image: Transformed image.
            label: Corresponding label for the image.
        """
        img_name = os.path.join(self.img_dir, self.metadata.iloc[idx]['filename'])
        label = self.metadata.iloc[idx]['label']

        # Load image and handle possible file errors
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_name}: {e}")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label

# Example usage of transforms
def get_data_transforms(augment=False):
    """
    Get train/validation data transforms.

    Args:
        augment (bool): Whether to apply data augmentation to training set.

    Returns:
        dict: Dictionary containing 'train' and 'val' transforms.
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return {
        'train': train_transform,
        'val': val_transform
    }
