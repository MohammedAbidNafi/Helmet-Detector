import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import os
from torchvision import models
import torch.nn as nn

# Load the trained model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust the output size for binary classification
model.load_state_dict(torch.load('helmet_classifier.pth'))
model.eval()

# Define a function to preprocess and classify an image
def classify_image(image):
    # Resize and preprocess the image to match the input requirements of the ResNet model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transformation to the image
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    # Use the trained model to make a prediction
    with torch.no_grad():
        outputs = model(image)

    # Get the predicted class (0 for "no helmet" and 1 for "with helmet")
    _, predicted_class = torch.max(outputs, 1)

    return predicted_class.item()

# Example usage for processing images in a directory
if __name__ == '__main__':
    # Directory containing the images
    image_dir = 'data/test'  # Change this to the directory containing test images

    # List all files in the directory
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpeg')]

    for image_path in image_files:
        image = Image.open(image_path)
        predicted_class = classify_image(image)

        if predicted_class == 1:
            print(f"\n \n The Rider in {image_path} is not wearing a helmet! \n \n")
        elif predicted_class == 0:
            print(f"\n \n The Rider in {image_path} is wearing a helmet! \n \n")
        elif predicted_class != 0 & predicted_class !=1:
            print(f"\n \n Not able to recognise what is happening here in {image_path}! \n \n")
