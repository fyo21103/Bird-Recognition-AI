import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import os

def load_data(data_dir, batch_size):
    # Define transformations for the test data
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the test data
    test_data = datasets.ImageFolder(root=data_dir, transform=test_transforms)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return test_loader

def load_model(model_path, num_classes):
    # Initialize and load the model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path))
    return model

def test_model(model, test_loader, device):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print('Test Accuracy: {:.2f}%'.format(accuracy))

def main():
    data_dir = 'C:/Users/o364b/source/repos/Bird-Recognition/Bird-Recognition-AI/test-data'  # Replace with the path to your test dataset
    model_path = 'C:/Users/o364b/source/repos/Bird-Recognition/Bird-Recognition-AI/trained-models/model.pth'  # Replace with the path to your trained model
    num_classes = 2  # Replace with the number of classes in your model
    batch_size = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_loader = load_data(data_dir, batch_size)
    model = load_model(model_path, num_classes)
    test_model(model, test_loader, device)

if __name__ == '__main__':
    main()
