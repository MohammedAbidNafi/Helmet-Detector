import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image, UnidentifiedImageError
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Define the custom dataset for image loading
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get the list of subdirectories (helmet and no_helmet)
        self.classes = os.listdir(root_dir)
        self.classes.sort()  # Ensure consistent class ordering

        # Initialize lists to store file paths and corresponding labels
        self.image_filenames = []
        self.labels = []

        # Iterate through subdirectories and collect image file paths and labels
        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue  # Skip if it's not a directory
            image_files = os.listdir(class_dir)
            image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            self.image_filenames.extend([os.path.join(class_dir, fname) for fname in image_files])
            self.labels.extend([i] * len(image_files))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]

        try:
            # Attempt to open the image
            image = Image.open(img_name)
        except (UnidentifiedImageError, OSError):
            raise ValueError(f"Failed to open image file '{img_name}'.")

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Set the data directory containing your training images
data_dir = 'data/train'
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory '{data_dir}' does not exist.")
print(f"Directory '{data_dir}' does exist!")

# Define data transformations for the model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a custom dataset and data loader
dataset = ImageDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))  # Adjust the output size for classification

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10  # Adjust the number of training epochs as needed

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

print("Training complete!")

# Save the trained model
torch.save(model.state_dict(), 'helmet_classifier.pth')
