import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision import models
import torch.nn as nn

# Load the trained model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)  # Adjust the output size for binary classification
model.load_state_dict(torch.load('helmet_classifier.pth'))
model.eval()

# Define a function to preprocess and classify an image
def classify_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)

    _, predicted_class = torch.max(outputs, 1)

    return predicted_class.item()

# Open a video capture object for the default camera (you can change the camera index if needed)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the OpenCV frame to a PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Classify the image
    predicted_class = classify_image(pil_image)

    # Display the result on the frame
    if predicted_class == 2:
        cv2.putText(frame, "No Helmet", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif predicted_class == 1:
        cv2.putText(frame, "With Helmet", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif predicted_class == 0:
        cv2.putText(frame, "Unknown", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Helmet Detection', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
