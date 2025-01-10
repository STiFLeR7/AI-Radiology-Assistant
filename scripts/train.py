import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
import pandas as pd
from chest_xray_dataset import ChestXrayDataset

# Paths
DATA_PATH = "D:/AI-Radiology-Assistant/data/raw/"
metadata = pd.read_csv(os.path.join(DATA_PATH, "metadata.csv"))
MODEL_PATH = "models/chest_xray_model.pth"
os.makedirs("models", exist_ok=True)

# Data Augmentation and Preprocessing
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Check and handle rare classes
print("Original class distribution:")
print(metadata['encoded_label'].value_counts())

if metadata['encoded_label'].value_counts().min() < 2:
    print("Some classes have less than 2 samples. Removing these classes...")
    metadata = metadata[metadata['encoded_label'].map(metadata['encoded_label'].value_counts()) > 1]

# Train-validation split
print("\nFiltered class distribution:")
print(metadata['encoded_label'].value_counts())

if metadata['encoded_label'].value_counts().min() < 2:
    print("Not enough samples for stratified split. Proceeding without stratification.")
    train_data, val_data = train_test_split(metadata, test_size=0.2, random_state=42)
else:
    train_data, val_data = train_test_split(metadata, test_size=0.2, stratify=metadata['encoded_label'], random_state=42)

# Datasets and DataLoaders
train_dataset = ChestXrayDataset(train_data, DATA_PATH, train_transform)
val_dataset = ChestXrayDataset(val_data, DATA_PATH, val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(metadata['encoded_label'].unique()))  # Adjust output layer
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
best_val_loss = float("inf")

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}\n{'-'*10}")
    
    # Training phase
    model.train()
    train_loss, train_correct = 0.0, 0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)
    
    train_loss /= len(train_loader.dataset)
    train_acc = train_correct.double() / len(train_loader.dataset)
    
    # Validation phase
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)
    
    val_loss /= len(val_loader.dataset)
    val_acc = val_correct.double() / len(val_loader.dataset)
    
    print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    
    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved to", MODEL_PATH)
