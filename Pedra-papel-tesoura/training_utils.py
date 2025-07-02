"""
Training utilities for Rock Paper Scissors detection
"""
import torch
import numpy as np
from collections import defaultdict
from config import Config

class Trainer:
    """Training utilities for the model"""
    
    def __init__(self, model, optimizer, device, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
    
    def train_one_epoch(self, data_loader):
        """Train model for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for images, targets in data_loader:
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            self.optimizer.step()
            
            total_loss += losses.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate(self, data_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in data_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Switch to training mode temporarily for loss calculation
                self.model.train()
                loss_dict = self.model(images, targets)
                self.model.eval()

                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train_model(self, train_loader, valid_loader):
        """Complete training loop"""
        print("\nStarting training...")
        print(f"Training on device: {self.device}")
        print(f"Number of epochs: {Config.NUM_EPOCHS}")
        print(f"Learning rate: {Config.LEARNING_RATE}")
        print("-" * 50)
        
        for epoch in range(Config.NUM_EPOCHS):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(valid_loader)
            
            print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print("-" * 50)
            
            # Log epoch metrics if logger is available
            if self.logger:
                self.logger.log_epoch_metrics(epoch, train_loss, val_loss)
            
            # Optional: Early stopping or learning rate scheduling could be added here
        
        print("Training completed!")

class Evaluator:
    """Model evaluation utilities"""
    
    def __init__(self, device):
        self.device = device
    
    def calculate_iou(self, box1, box2):
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
    
    def evaluate_model(self, model, valid_loader):
        """Evaluate the trained model"""
        print("\nEvaluating model...")
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in valid_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                predictions = model(images)
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Calculate metrics
        metrics = defaultdict(list)
        
        for pred, target in zip(all_predictions, all_targets):
            pred_boxes = pred['boxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            
            target_boxes = target['boxes'].cpu().numpy()
            target_labels = target['labels'].cpu().numpy()
            
            for pred_box, pred_score, pred_label in zip(pred_boxes, pred_scores, pred_labels):
                if pred_score > Config.CONFIDENCE_THRESHOLD:
                    best_iou = 0
                    
                    for target_box, target_label in zip(target_boxes, target_labels):
                        if pred_label == target_label:
                            iou = self.calculate_iou(pred_box, target_box)
                            if iou > best_iou:
                                best_iou = iou
                    
                    for threshold in Config.IOU_THRESHOLDS:
                        if best_iou >= threshold:
                            metrics[f'correct_{threshold}'].append(1)
                        else:
                            metrics[f'correct_{threshold}'].append(0)
                    
                    metrics['iou_scores'].append(best_iou)
        
        # Print results
        if metrics['iou_scores']:
            avg_iou = np.mean(metrics['iou_scores'])
            print(f"Average IoU: {avg_iou:.4f}")
            
            for threshold in Config.IOU_THRESHOLDS:
                if f'correct_{threshold}' in metrics:
                    precision = np.mean(metrics[f'correct_{threshold}'])
                    print(f"Precision at IoU {threshold}: {precision:.4f}")
        else:
            print("No predictions above confidence threshold found.")
        
        print("Evaluation completed!")
        return metrics
