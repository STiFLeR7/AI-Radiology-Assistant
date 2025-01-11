import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from chest_xray_dataset import ChestXrayDataset  # Dataset class

# Paths
MODEL_PATH = "D:/AI-Radiology-Assistant/models/resnet18_chest_xray.pth"
DATASET_PATH = "D:/AI-Radiology-Assistant/data/raw/"
METADATA_PATH = os.path.join(DATASET_PATH, "metadata.csv")
IMAGE_DIR = os.path.join(DATASET_PATH, "images")
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load Dataset
def load_dataset(metadata_path, image_dir, transform):
    metadata = pd.read_csv(metadata_path)
    dataset = ChestXrayDataset(
        image_dir=image_dir,
        metadata=metadata,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return dataloader, dataset

# Load Model
def load_model(model_path, num_classes):
    from torchvision.models import resnet18
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Evaluate Model
def evaluate_model(model, dataloader, num_classes):
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    class_names = dataloader.dataset.metadata['encoded_label'].unique()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i]
                if predicted[i] == label:
                    class_correct[label] += 1
                class_total[label] += 1

    overall_accuracy = correct / total * 100
    class_accuracies = {class_names[i]: class_correct[i] / class_total[i] * 100
                        for i in range(num_classes)}

    return overall_accuracy, class_accuracies

# Main evaluation logic
if __name__ == "__main__":
    # Load validation dataset
    val_loader, val_dataset = load_dataset(METADATA_PATH, IMAGE_DIR, data_transform)
    num_classes = len(val_dataset.metadata['encoded_label'].unique())

    # Load trained model
    model = load_model(MODEL_PATH, num_classes)

    # Evaluate the model
    overall_accuracy, class_accuracies = evaluate_model(model, val_loader, num_classes)

    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    for class_name, acc in class_accuracies.items():
        print(f"Accuracy for class '{class_name}': {acc:.2f}%")
