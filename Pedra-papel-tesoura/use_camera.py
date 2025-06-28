import cv2
import torch
import numpy as np
from PIL import Image
import time
from torchvision import transforms
import joblib

def get_transform(train):
    """Get image transforms for training or validation"""
    transform_list = [
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
    ]
    if train:
        transform_list.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transform_list)

def webcam_detection(model, device, class_names):
    """
    Real-time object detection using webcam
    """
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam started. Press 'q' to quit.")
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Transform image
        transform = get_transform(train=False)
        image_tensor = transform(pil_image)  # Remove unsqueeze here since transform already returns a tensor
        
        # Get prediction
        with torch.no_grad():
            prediction = model([image_tensor.to(device)])[0]  # Pass as list of tensors
        
        # Draw predictions on frame
        for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
            if score > 0.5:  # Only show high confidence predictions
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label and confidence
                label_text = f"{class_names[label]} ({score:.2f})"
                cv2.putText(frame, label_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Rock Paper Scissors Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Load the trained model and setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = joblib.load('trained_model.joblib')
model.to(device)
model.eval()

# Define class names
class_names = ['rock', 'paper', 'scissors']

# Start real-time detection
print("Starting real-time webcam detection...")
webcam_detection(model, device, class_names)
