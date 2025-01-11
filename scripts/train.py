import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from chest_xray_dataset import ChestXrayDataset
import pandas as pd
from tqdm import tqdm

# Paths
DATASET_PATH = "D:/AI-Radiology-Assistant/data/raw/"
METADATA_PATH = os.path.join(DATASET_PATH, "metadata.csv")
IMAGE_DIR = os.path.join(DATASET_PATH, "images")
MODEL_PATH = "D:/AI-Radiology-Assistant/models/resnet18_chest_xray.pth"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Dataset and Dataloader
def load_data():
    metadata = pd.read_csv(METADATA_PATH)

    # Add encoded_label if it doesn't exist
    if "encoded_label" not in metadata.columns:
        label_mapping = {label: idx for idx, label in enumerate(metadata["finding"].unique())}
        metadata["encoded_label"] = metadata["finding"].map(label_mapping)
        print("Original Label Mapping:", label_mapping)

    # Filter out classes with fewer than 2 samples
    class_counts = metadata["encoded_label"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    metadata = metadata[metadata["encoded_label"].isin(valid_classes)]
    print(f"Filtered metadata: {len(metadata)} samples, {len(valid_classes)} classes remaining.")

    # Recompute encoded labels to ensure they are continuous
    valid_labels = metadata["finding"].unique()
    new_label_mapping = {label: idx for idx, label in enumerate(valid_labels)}
    metadata["encoded_label"] = metadata["finding"].map(new_label_mapping)
    print("Updated Label Mapping:", new_label_mapping)

    # Debug: Ensure all encoded labels are within range
    num_classes = len(new_label_mapping)
    assert metadata["encoded_label"].min() >= 0, "Found negative encoded labels!"
    assert metadata["encoded_label"].max() < num_classes, "Encoded labels exceed num_classes!"

    # Train-test split
    train_metadata, val_metadata = train_test_split(
        metadata, test_size=0.2, stratify=metadata["encoded_label"], random_state=42
    )

    train_dataset = ChestXrayDataset(
        image_dir=IMAGE_DIR, metadata=train_metadata, transform=train_transform
    )
    val_dataset = ChestXrayDataset(
        image_dir=IMAGE_DIR, metadata=val_metadata, transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, num_classes


# Model Training Function
def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10):
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Debug: Check label range
            assert torch.all((labels >= 0) & (labels < model.fc.out_features)), (
                f"Invalid label found in batch! Labels: {labels}"
            )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Main Training Logic
if __name__ == "__main__":
    train_loader, val_loader, num_classes = load_data()

    # Define Model
    from torchvision.models import resnet18
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the Model
    train_model(model, criterion, optimizer, train_loader, val_loader, DEVICE, NUM_EPOCHS)
