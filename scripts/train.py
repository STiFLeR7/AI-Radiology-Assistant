import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from chest_xray_dataset import ChestXrayDataset

# Paths
dataset_dir = "D:/AI-Radiology-Assistant/data/raw/"
image_dir = os.path.join(dataset_dir, "images")
metadata_path = os.path.join(dataset_dir, "metadata.csv")

# Load metadata
metadata = pd.read_csv(metadata_path)

# Debugging metadata loading
print("Metadata type:", type(metadata))  # Should be <class 'pandas.core.frame.DataFrame'>
print(metadata.head())  # Display the first few rows of metadata

# Encode labels
if 'encoded_label' not in metadata.columns:
    label_mapping = {label: idx for idx, label in enumerate(metadata['finding'].unique())}
    metadata['encoded_label'] = metadata['finding'].map(label_mapping)

# Debug label mapping
print("Label Mapping:", label_mapping)
print("Sample Encoded Labels:", metadata[['finding', 'encoded_label']].head())

# Filter out rare classes
class_counts = metadata['encoded_label'].value_counts()
filtered_classes = class_counts[class_counts >= 2].index  # Keep classes with at least 2 samples
metadata = metadata[metadata['encoded_label'].isin(filtered_classes)]

# Debug class distribution
print("Filtered class distribution:", metadata['encoded_label'].value_counts())

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset and DataLoader
train_dataset = ChestXrayDataset(image_dir=image_dir, metadata=metadata, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Debugging dataset
print("Dataset initialized. Sample batch:")
for inputs, labels in train_loader:
    print("Inputs shape:", inputs.shape)
    print("Labels:", labels)
    break

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights="IMAGENET1K_V1")  # Updated to new 'weights' argument
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(label_mapping))  # Adjust final layer to match number of classes
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 10)
    model.train()
    
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    print(f"Loss: {epoch_loss:.4f}")

# Save the model
model_save_path = "D:/AI-Radiology-Assistant/models/resnet18_chest_xray.pth"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
