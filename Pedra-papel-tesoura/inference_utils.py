"""
Inference utilities for Rock Paper Scissors detection
"""
import torch
import cv2
import numpy as np
import pandas as pd
import os
import glob
from PIL import Image
from torchvision import transforms as T
from config import Config
from collections import defaultdict




class InferenceEngine:
    """Inference engine for Rock Paper Scissors detection"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.transform = T.Compose([T.ToTensor()])
        self.model.eval()
    
    def predict_image(self, image_path, show_results=True):
        """Predict on a single image"""
        try:
            # Load and preprocess image
            img = Image.open(image_path)
            img_tensor = self.transform(img)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Model prediction
            with torch.no_grad():
                prediction = self.model(img_tensor)
            
            # Process results
            boxes = prediction[0]['boxes'].cpu().numpy()
            labels = prediction[0]['labels'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()
            
            if show_results:
                print(f"\nPredictions for {image_path}:")
                print("-" * 50)
                
                found_predictions = False
                for box, label, score in zip(boxes, labels, scores):
                    if score > Config.CONFIDENCE_THRESHOLD:
                        class_name = Config.CLASS_NAMES[label] if label < len(Config.CLASS_NAMES) else f"Class_{label}"
                        print(f"Class: {class_name}, Score: {score:.3f}, Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
                        found_predictions = True
                
                if not found_predictions:
                    print("No high-confidence predictions found.")
            
            return boxes, labels, scores
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None, None, None
    
    def predict_validation_images(self, validation_images):
        """Run inference on all validation images"""
        print(f"\nRunning inference on {len(validation_images)} validation images...")
        print("=" * 60)
        
        results = []
        for i, image_path in enumerate(validation_images):
            print(f"\nProcessing image {i+1}/{len(validation_images)}: {image_path}")
            boxes, labels, scores = self.predict_image(image_path, show_results=True)
            
            if boxes is not None:
                results.append({
                    'image_path': image_path,
                    'boxes': boxes,
                    'labels': labels,
                    'scores': scores
                })
        
        print(f"\nInference completed on {len(results)} images successfully.")
        return results

class WebcamDetection:
    """Real-time webcam detection"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def get_transform(self, train=False):
        """Get image transforms for webcam frames"""
        transform_list = [
            T.Resize(Config.IMAGE_SIZE),
            T.ToTensor(),
        ]
        return T.Compose(transform_list)
    
    def run_webcam_detection(self):
        """Run real-time detection using webcam"""
        print("\nStarting webcam detection...")
        print(f"Running on device: {self.device}")
        print("Press 'q' to quit.")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set webcam properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        transform = self.get_transform(train=False)
        
        print("Webcam started successfully!")
        print("=" * 50)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Preprocess image
                image_tensor = transform(pil_image).unsqueeze(0).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    prediction = self.model(image_tensor)[0]
                
                # Draw predictions
                for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
                    if score > Config.CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                        
                        # Scale coordinates back to original frame size
                        height, width = frame.shape[:2]
                        x1 = int(x1 * width / Config.IMAGE_SIZE[0])
                        y1 = int(y1 * height / Config.IMAGE_SIZE[1])
                        x2 = int(x2 * width / Config.IMAGE_SIZE[0])
                        y2 = int(y2 * height / Config.IMAGE_SIZE[1])
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        class_name = Config.CLASS_NAMES[label] if label < len(Config.CLASS_NAMES) else f"Class_{label}"
                        label_text = f"{class_name} ({score:.2f})"
                        cv2.putText(frame, label_text, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Rock Paper Scissors Detection', frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nWebcam detection interrupted by user.")
        except Exception as e:
            print(f"\nError during webcam detection: {str(e)}")
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("Webcam detection stopped.")
    
    def test_webcam_access(self):
        """Test if webcam is accessible"""
        print("Testing webcam access...")
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("Webcam test successful!")
                cap.release()
                return True
            else:
                print("Webcam opened but cannot read frames.")
                cap.release()
                return False
        else:
            print("Cannot open webcam.")
            return False

class ModelEvaluator:
    """Evaluate model predictions against ground truth annotations"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.inference_engine = InferenceEngine(model, device)
    
    def load_ground_truth(self, annotations_path):
        """Load ground truth annotations from CSV file"""
        try:
            df = pd.read_csv(annotations_path)
            print(f"Loaded {len(df)} ground truth annotations from {annotations_path}")
            return df
        except Exception as e:
            print(f"Error loading annotations: {str(e)}")
            return None
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes"""
        # Box format: [xmin, ymin, xmax, ymax]
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmin >= inter_xmax or inter_ymin >= inter_ymax:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def evaluate_model(self, test_images_path, annotations_path):
        """Evaluate model against ground truth annotations"""
        print("\n" + "="*60)
        print("MODEL EVALUATION AGAINST GROUND TRUTH")
        print("="*60)
        
        # Load ground truth
        ground_truth_df = self.load_ground_truth(annotations_path)
        if ground_truth_df is None:
            return None
        
        # Get list of test images
        test_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend(glob.glob(os.path.join(test_images_path, ext)))
        
        if not test_images:
            print(f"No test images found in {test_images_path}")
            return None
        
        print(f"Found {len(test_images)} test images")
        
        # Evaluation metrics
        total_predictions = 0
        correct_predictions = 0
        class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        # Process each image
        for image_path in test_images:
            image_filename = os.path.basename(image_path)
            
            # Get ground truth for this image
            gt_rows = ground_truth_df[ground_truth_df['filename'] == image_filename]
            if gt_rows.empty:
                print(f"No ground truth found for {image_filename}, skipping...")
                continue
            
            # Get model predictions
            boxes, labels, scores = self.inference_engine.predict_image(image_path, show_results=False)
            if boxes is None:
                continue
            
            # Filter predictions by confidence threshold
            high_conf_mask = scores > Config.CONFIDENCE_THRESHOLD
            pred_boxes = boxes[high_conf_mask]
            pred_labels = labels[high_conf_mask]
            pred_scores = scores[high_conf_mask]
            
            # Match predictions with ground truth
            matched_gt = set()
            
            for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
                total_predictions += 1
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching ground truth
                for idx, (_, gt_row) in enumerate(gt_rows.iterrows()):
                    if idx in matched_gt:
                        continue
                    
                    gt_box = [gt_row['xmin'], gt_row['ymin'], gt_row['xmax'], gt_row['ymax']]
                    gt_class = gt_row['class']
                    
                    # Convert class name to label index
                    if gt_class in Config.CLASS_NAMES:
                        gt_label = Config.CLASS_NAMES.index(gt_class)
                    else:
                        continue
                    
                    iou = self.calculate_iou(pred_box, gt_box)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                        best_gt_label = gt_label
                
                # Check if prediction is correct
                pred_class_name = Config.CLASS_NAMES[pred_label] if pred_label < len(Config.CLASS_NAMES) else f"Class_{pred_label}"
                
                if best_iou >= 0.5 and pred_label == best_gt_label:  # IoU threshold of 0.5
                    correct_predictions += 1
                    matched_gt.add(best_gt_idx)
                    class_stats[pred_class_name]['tp'] += 1
                    print(f"✓ CORRECT: {image_filename} - {pred_class_name} (IoU: {best_iou:.3f}, Score: {pred_score:.3f})")
                else:
                    class_stats[pred_class_name]['fp'] += 1
                    if best_gt_label is not None:
                        gt_class_name = Config.CLASS_NAMES[best_gt_label]
                        print(f"✗ WRONG: {image_filename} - Predicted: {pred_class_name}, Actual: {gt_class_name} (IoU: {best_iou:.3f}, Score: {pred_score:.3f})")
                    else:
                        print(f"✗ FALSE POSITIVE: {image_filename} - {pred_class_name} (Score: {pred_score:.3f})")
            
            # Count missed ground truth (false negatives)
            for idx, (_, gt_row) in enumerate(gt_rows.iterrows()):
                if idx not in matched_gt:
                    gt_class = gt_row['class']
                    if gt_class in Config.CLASS_NAMES:
                        class_stats[gt_class]['fn'] += 1
                        print(f"✗ MISSED: {image_filename} - {gt_class} not detected")
        
        # Calculate and display metrics
        self.display_evaluation_results(total_predictions, correct_predictions, class_stats)
        
        return {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0,
            'class_stats': dict(class_stats)
        }
    
    def display_evaluation_results(self, total_predictions, correct_predictions, class_stats):
        """Display detailed evaluation results"""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Overall accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Overall Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        print("-" * 50)
        print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 50)
        
        for class_name in Config.CLASS_NAMES[1:]:  # Skip background class
            if class_name in class_stats:
                stats = class_stats[class_name]
                tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"{class_name:<12} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")
            else:
                print(f"{class_name:<12} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
        
        print("-" * 50)
