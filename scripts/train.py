import os
import torch
import pandas as pd
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
from collections import Counter
from PIL import Image


class ChestXrayDataset(Dataset):
    """
    Custom dataset for loading Chest X-ray images and their labels.
    """
    def __init__(self, metadata_csv, img_dir, transform=None):
        self.metadata = pd.read_csv(metadata_csv)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        
        # Convert the label to a PyTorch LongTensor
        label = torch.tensor(row['label'], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        return image, label


def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10):
    """
    Train the model and validate on the validation dataset.
    """
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    return model


# Main Execution
if __name__ == "__main__":
    # Paths
    metadata_csv = "data/processed/processed_metadata.csv"
    img_dir = "data/processed/images"
    model_save_path = "models/chest_xray_model.pth"

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load metadata
    metadata = pd.read_csv(metadata_csv)

    # Create a 'label' column from 'finding'
    metadata['label'] = metadata['finding'].astype('category').cat.codes
    metadata.to_csv(metadata_csv, index=False)  # Save updated metadata

    # Dataset and DataLoader
    dataset = ChestXrayDataset(metadata_csv, img_dir, transform)

    # Filter Rare Classes (classes with < 2 samples)
    class_counts = Counter(dataset.metadata['label'])
    valid_labels = [label for label, count in class_counts.items() if count > 1]
    dataset.metadata = dataset.metadata[dataset.metadata['label'].isin(valid_labels)].reset_index(drop=True)

    if len(valid_labels) < 2:
        raise ValueError("Not enough classes with sufficient samples to perform training.")

    # Update dataset length and relabel indices
    dataset.metadata['label'] = dataset.metadata['label'].astype('category').cat.codes

    # Train-Test Split
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        stratify=dataset.metadata['label'],
        random_state=42
    )

    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_indices),
        batch_size=16,
        shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_indices),
        batch_size=16,
        shuffle=False
    )

    # Model
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(dataset.metadata['label'].unique()))  # Adjust output layer for classes
    model = model.to(device)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train Model
    model = train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10)

    # Save Model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
