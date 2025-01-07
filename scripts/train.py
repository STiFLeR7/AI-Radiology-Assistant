import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# Dataset Class
class ChestXrayDataset(Dataset):
    def __init__(self, metadata_csv, img_dir, transform=None):
        self.metadata = pd.read_csv(metadata_csv)
        self.img_dir = img_dir
        self.transform = transform

        # Map the 'finding' column to numeric labels
        self.metadata['label'] = self.metadata['finding'].astype('category').cat.codes
        self.label_mapping = dict(enumerate(self.metadata['finding'].astype('category').cat.categories))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.metadata.iloc[idx]['filename'])
        image = Image.open(img_path).convert("RGB")
        label = self.metadata.iloc[idx]['label']  # Numeric label for the finding

        if self.transform:
            image = self.transform(image)

        return image, label

# Training Function
def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Training Loss: {running_loss / len(train_loader)}")

        # Validation phase
        model.eval()
        val_accuracy = 0.0
        val_predictions = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_predictions.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_predictions)
        print(f"Validation Accuracy: {val_accuracy}")

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

    # Dataset and DataLoader
    dataset = ChestXrayDataset(metadata_csv, img_dir, transform)
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
