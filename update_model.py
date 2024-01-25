import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from identification import ResNet34  

# Define the device to use: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model with the number of classes it was originally trained with
num_original_classes = 450  # Change to the number of classes the pre-trained model has
model = ResNet34(in_channels=3, num_classes=num_original_classes).to(device)

# Load the state dictionary
state_dict_path = 'trained-models/bird-resnet34best.pth'
state_dict = torch.load(state_dict_path, map_location=device)

# If the state dict has 'network.' as a prefix for all keys (which seems to be the case)
# Remove this prefix
adjusted_state_dict = {k.replace('network.', ''): v for k, v in state_dict.items()}

# Load the adjusted state dictionary into the model
model.load_state_dict(adjusted_state_dict)

# If the number of classes in the new task is different, adjust the classifier accordingly
num_new_classes = 500  # Set to the number of classes for the new task
if num_new_classes != num_original_classes:
    # Replace the last layer with a new one (it will have requires_grad=True by default)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_new_classes)

# Move the model to the device after making modifications
model = model.to(device)

# Define your loss function and optimizer
# The optimizer should only be constructed after the model is moved to the device
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define transformations for your image data
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your training and validation data
train_data = ImageFolder(root='birds_dataset/train', transform=transform)
val_data = ImageFolder(root='birds_dataset/val', transform=transform)

# Create DataLoaders for your datasets
# Set num_workers according to your system's specifications
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)

# A training function
def train_model(model, criterion, optimizer, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Move inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            # print("Inputs device: ", inputs.device)
            # print("Labels device: ", labels.device)
            # print("Outputs device: ", outputs.device)
            # print("Model device: ", next(model.parameters()).device)


            # Compute loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')

    print('Finished Training')
    return model

def calculate_accuracy(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Deactivate autograd to reduce memory usage and speed up computations
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Main block
if __name__ == "__main__":
    trained_model = train_model(model, criterion, optimizer, num_epochs=50)
    # Save the entire model
    torch.save(model, 'trained-models/new_update.pth')
    train_accuracy = calculate_accuracy(model, train_loader, device)
    print(f'Training Accuracy: {train_accuracy}%')


    # Add inference code as needed