import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Define Dataset
# Define Dataset
class BraTSDataset(Dataset):
    def __init__(self, metadata_csv, transform=None):
        """
        Initialize the dataset.

        Args:
            metadata_csv (str): Path to the metadata CSV file.
            transform (callable, optional): Optional transform to be applied to each sample.
        """
        self.data = pd.read_csv(metadata_csv)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single data point.

        Args:
            idx (int): Index of the data point.

        Returns:
            tuple: (image, target) where image is the input image tensor and target is the label tensor.
        """
        row = self.data.iloc[idx]
        image_path = row['slice_path']  # Adjust column name if necessary
        target = row['target']

        # Load the image (assuming .npy format for slices)
        image = np.load(image_path)  # Replace with appropriate image loading if needed
        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32).unsqueeze(0), torch.tensor(target, dtype=torch.long)


# Define Model (example: simple CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)  # Adjust based on your image dimensions
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training Function
def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_acc = 100. * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

    return model

# Main
if __name__ == "__main__":
    metadata_csv = "D:/AI-Radiology-Assistant/data/processed/mapped_metadata.csv"
    save_model_path = "D:/AI-Radiology-Assistant/models/best_model.pth"
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset and Dataloaders
    dataset = BraTSDataset(metadata_csv)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the Model
    trained_model = train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, device)

    # Save the Model
    torch.save(trained_model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")
