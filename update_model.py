import torch
from torchvision import models
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim

from identification import ResNet34

# Define the device to use: GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = ResNet34(3, 450)  # Use the actual number of classes
model = model.to(device)

# Load the state dictionary
state_dict = torch.load('trained-models/bird-resnet34best.pth')



# Customize the fully connected layer (fc) to the number of classes you have
# Replace num_classes with the actual number of bird classes
num_classes = 10  # Example: 10 different bird species
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

# Update the model's state dictionary
model.load_state_dict(state_dict, strict=False)

# Put the model in evaluation mode if you're not training
# For inference/prediction only
# model.eval()

# If you are planning to train the model further, keep it in training mode
# And define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define transformations for your image data
transform = Compose([
    Resize((224, 224)),  # Assuming you are using the standard image size for ResNet
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your training and validation data
train_data = ImageFolder(root='birds_dataset/train', transform=transform)
val_data = ImageFolder(root='birds_dataset/val', transform=transform)

# Create DataLoaders for your datasets
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)

# A training function (if you plan to train the model further)
def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)  # Move inputs to the device
            labels = labels.to(device)  # Move labels to the device

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Now both tensors are on the same device
            loss.backward()
            optimizer.step()


            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')
    
    print('Finished Training')
    return model


# The main block to either train the model or make predictions
if __name__ == "__main__":
    # Call the training function
    trained_model = train_model(model, criterion, optimizer, num_epochs=25)

    # Or perform inference on a single batch of data
    # model.eval()  # Set the model to evaluation mode
    # with torch.no_grad():
    #    for inputs, labels in val_loader:
    #        # Move inputs to the correct device
    #        inputs = inputs.to(device)
    #        outputs = model(inputs)
    # Continue with your inference process

