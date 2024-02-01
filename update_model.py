import os
import pandas as pd
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.nn import functional as F
from identification import ResNet34  # Ensure this import is correct
import torch.nn as nn

class WrappedResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = ResNet34(3, 450)

    def forward(self, x):
        return self.network(x)

def main():
    # Paths to your new train and validation folders
    new_train_data_dir = 'birds_dataset/train'
    new_val_data_dir = 'birds_dataset/val'

    # Replace these with your calculated mean and standard deviation
    mean = [0.485, 0.456, 0.406]  # Example values
    std = [0.229, 0.224, 0.225]   # Example values

    # Define transformations
    train_tfms = T.Compose([
        T.Resize(256),  # Resize the image so it's at least 256x256
        T.RandomCrop(224, padding=4, padding_mode='reflect'),
        T.RandomApply(torch.nn.ModuleList([T.GaussianBlur(kernel_size=3, sigma=(0.2, 5))]), p=0.15),
        T.RandomHorizontalFlip(), 
        T.RandomRotation(10),
        T.ToTensor(), 
        T.Normalize(mean, std, inplace=True)
    ])

    val_tfms = T.Compose([
        T.Resize(224),  # Resize images to 224x224 for validation
        T.ToTensor(), 
        T.Normalize(mean, std, inplace=True)
    ])

    # Load new datasets
    train_dataset = ImageFolder(new_train_data_dir, transform=train_tfms)
    val_dataset = ImageFolder(new_val_data_dir, transform=val_tfms)

    # Create DataLoaders for new datasets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Initialize the wrapped model and load your existing model
    wrapped_model = WrappedResNet34()
    try:
        wrapped_model.load_state_dict(torch.load('trained-models/bird-resnet34best.pth'))
        print("Model loaded successfully")
    except RuntimeError as e:
        print("Error in loading model: ", e)
        return  # Exit if model loading fails

    # Extract the original ResNet34 model
    model = wrapped_model.network

    # Transfer model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Training settings - adjust these parameters as needed
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 25  # Number of epochs for retraining

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Validation step
        model.eval()
        val_loss_total = 0
        num_batches = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss_total += F.cross_entropy(outputs, labels).item()
            num_batches += 1
        val_loss_avg = val_loss_total / num_batches
        print(f'Validation Loss: {val_loss_avg:.4f}')

    # Save the updated model
    torch.save(model.state_dict(), 'trained-models/updated_model.pth')

    print("Model updated and saved successfully.")

if __name__ == '__main__':
    main()
