import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from collections import defaultdict
import joblib
import matplotlib.patches as patches
from PIL import Image
import time

def main():
    """Main function for Rock Paper Scissors object detection"""
    
    # Load and examine the dataset
    print("Loading dataset...")
    df_image_info = pd.read_csv('./test/_annotations.csv')
    print(f"Dataset shape: {df_image_info.shape}")
    print(df_image_info.head())
    print(df_image_info.describe())
    
    # Data Quality Check
    print("\n" + "="*50)
    print("DATA QUALITY CHECK")
    print("="*50)
    
    # Check for duplicate rows
    duplicate_rows = df_image_info.duplicated()
    print(f"Number of duplicate rows: {duplicate_rows.sum()}")
    
    if duplicate_rows.sum() > 0:
        print("\nDuplicate rows found:")
        print(df_image_info[duplicate_rows])
    else:
        print("No duplicate rows found.")
    
    # Check for duplicate filenames
    duplicate_filenames = df_image_info['filename'].duplicated()
    print(f"\nNumber of duplicate filenames: {duplicate_filenames.sum()}")
    
    if duplicate_filenames.sum() > 0:
        print("\nDuplicate filenames:")
        print(df_image_info[duplicate_filenames]['filename'].values)
    
    # Check for any missing values
    print(f"\nMissing values per column:")
    print(df_image_info.isnull().sum())
    
    # Check for invalid bounding box coordinates
    invalid_boxes = df_image_info[(df_image_info['xmin'] >= df_image_info['xmax']) | 
                                 (df_image_info['ymin'] >= df_image_info['ymax'])]
    print(f"\nNumber of invalid bounding boxes: {len(invalid_boxes)}")
    
    if len(invalid_boxes) > 0:
        print("Invalid bounding boxes found:")
        print(invalid_boxes)
    
    # Visualize class distribution
    print("\nVisualizing class distribution...")
    plt.style.use('_mpl-gallery')
    class_counts = df_image_info['class'].value_counts()
    
    plt.figure(figsize=(8, 6))
    plt.bar(class_counts.index, class_counts.values, alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Distribution of Rock, Paper, Scissors Classes')
    plt.grid(True, alpha=0.3)
    # plt.show()
    
    # Model setup - Choose to load existing model or train new one
    model_filename = 'trained_model.joblib'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(model_filename):
        # Load existing model
        print(f"\nFound existing model at {model_filename}")
        choice = input("Do you want to load the existing model? (y/n): ").lower().strip()
        
        if choice == 'y' or choice == 'yes':
            print("Loading existing model...")
            model = joblib.load(model_filename)
            model = model.to(device)
            print("Model loaded successfully!")
        else:
            # Train new model
            model = train_new_model(df_image_info, device, model_filename)
    else:
        # No existing model found, train new one
        print(f"\nNo existing model found at {model_filename}")
        print("Training new model...")
        model = train_new_model(df_image_info, device, model_filename)
    
    # Start real-time detection
    print("\nStarting real-time webcam detection...")
    class_names = ['Background', 'Rock', 'Paper', 'Scissors']
    # webcam_detection(model, device, class_names)


def train_new_model(df_image_info, device, model_filename):
    """Train and save a new model"""
    # Load and process images
    print("\nLoading and processing images...")
    transform = T.Compose([T.ToTensor()])
    
    image_tensors = []
    targets = []
    
    for i in range(len(df_image_info)):
        image_path = f"./test/{df_image_info.loc[i, 'filename']}"
        img = cv2.imread(image_path)
        
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            x1 = df_image_info.loc[i, 'xmin']
            y1 = df_image_info.loc[i, 'ymin']
            x2 = df_image_info.loc[i, 'xmax']
            y2 = df_image_info.loc[i, 'ymax']
            
            boxes = [[x1, y1, x2, y2]]
            labels = [1]  # Placeholder label
            
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64)
            }
            
            img_tensor = transform(img)
            image_tensors.append(img_tensor)
            targets.append(target)
    
    print(f"Loaded {len(image_tensors)} images successfully")
    
    # Setup model
    print("\nSetting up model...")
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    num_classes = 4  # 3 classes (Rock, Paper, Scissors) + background
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Create dataset and data loaders
    print("Creating data loaders...")
    dataset = SimpleDataset(image_tensors, targets)
    
    indices = torch.randperm(len(dataset)).tolist()
    train_dataset = torch.utils.data.Subset(dataset, indices[:-50])
    valid_dataset = torch.utils.data.Subset(dataset, indices[-50:])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, 
                             collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, 
                             collate_fn=lambda x: tuple(zip(*x)))
    
    # Training
    print("\nStarting training...")
    num_epochs = 30
    learning_rate = 0.005
    
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    print(f"Training on device: {device}")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss = validate(model, valid_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print("-" * 50)
    
    # Save model
    print("\nSaving model...")
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, valid_loader, device)
    
    return model



class SimpleDataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


def train_one_epoch(model, optimizer, data_loader, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    
    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    return total_loss / len(data_loader)


def validate(model, data_loader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            model.train()
            loss_dict = model(images, targets)
            model.eval()

            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    
    return total_loss / len(data_loader)


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0


def evaluate_model(model, valid_loader, device):
    """Evaluate the trained model"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in valid_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            predictions = model(images)
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    # Calculate metrics
    iou_thresholds = [0.5, 0.75]
    class_names = ['Rock', 'Paper', 'Scissors']
    metrics = defaultdict(list)
    
    for pred, target in zip(all_predictions, all_targets):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        
        target_boxes = target['boxes'].cpu().numpy()
        target_labels = target['labels'].cpu().numpy()
        
        for i, (pred_box, pred_score, pred_label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            best_iou = 0
            best_match = None
            
            for j, (target_box, target_label) in enumerate(zip(target_boxes, target_labels)):
                if pred_label == target_label:
                    iou = calculate_iou(pred_box, target_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = j
            
            for threshold in iou_thresholds:
                if best_iou >= threshold:
                    metrics[f'correct_{threshold}'].append(1)
                else:
                    metrics[f'correct_{threshold}'].append(0)
            
            metrics['iou_scores'].append(best_iou)
    
    # Print evaluation results
    print("Model Evaluation Results:")
    print("=" * 50)
    
    avg_iou = np.mean(metrics['iou_scores'])
    print(f"Average IoU: {avg_iou:.4f}")
    
    for threshold in iou_thresholds:
        precision = np.mean(metrics[f'correct_{threshold}'])
        print(f"Precision at IoU {threshold}: {precision:.4f}")


def get_transform(train):
    """Get image transforms for training or validation"""
    transform_list = [
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
    ]
    if train:
        transform_list.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(transform_list)


# def webcam_detection(model, device, class_names):
#     """Real-time object detection using webcam"""
#     cap = cv2.VideoCapture(0)
    
#     if not cap.isOpened():
#         print("Error: Could not open webcam")
#         return
    
#     print("Webcam started. Press 'q' to quit.")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame")
#             break
        
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         pil_image = Image.fromarray(frame_rgb)
        
#         transform = get_transform(train=False)
#         image_tensor = transform(pil_image).unsqueeze(0)
        
#         with torch.no_grad():
#             prediction = model([image_tensor.to(device)])[0]
        
#         for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
#             if score > 0.5:
#                 x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
#                 label_text = f"{class_names[label]} ({score:.2f})"
#                 cv2.putText(frame, label_text, (x1, y1-10), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         cv2.imshow('Rock Paper Scissors Detection', frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
