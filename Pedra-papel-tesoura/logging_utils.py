"""
Logging utilities for Rock Paper Scissors detection
Records comprehensive execution logs including training metrics, device info, and timing
"""
import os
import json
import time
from datetime import datetime
import torch
import numpy as np
from config import Config

class ExecutionLogger:
    """Comprehensive logging system for training and inference runs"""
    
    def __init__(self, num_epochs=None):
        """Initialize logger with unique folder based on epochs and timestamp"""
        self.start_time = time.time()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create unique log folder name
        if num_epochs:
            self.log_folder = f"logs/{num_epochs}epochs_{self.timestamp}"
        else:
            self.log_folder = f"logs/inference_{self.timestamp}"
        
        # Create log directory
        os.makedirs(self.log_folder, exist_ok=True)
        
        # Initialize log data structure
        self.log_data = {
            "execution_info": {
                "start_time": datetime.now().isoformat(),
                "timestamp": self.timestamp,
                "num_epochs": num_epochs,
                "execution_type": "training" if num_epochs else "inference"
            },
            "system_info": {},
            "data_info": {},
            "model_info": {},
            "training_metrics": {},
            "validation_metrics": {},
            "test_metrics": {},
            "inference_results": {},
            "timing": {}
        }
        
        # Log system information immediately
        self._log_system_info()
        
        print(f"📝 Logging enabled - Results will be saved to: {self.log_folder}")
    
    def _convert_to_serializable(self, obj):
        """Convert non-serializable objects to JSON-serializable types"""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif torch.is_tensor(obj):
            return obj.detach().cpu().numpy().tolist() if obj.numel() > 1 else float(obj.detach().cpu().numpy())
        elif hasattr(obj, 'item'):  # For single-element tensors
            return obj.item()
        else:
            return obj
    
    def _log_system_info(self):
        """Log system and device information"""
        device = Config.get_device()
        
        system_info = {
            "device_type": device.type,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            system_info.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                "num_gpus": torch.cuda.device_count()
            })
        
        # Add configuration parameters
        system_info.update({
            "num_classes": Config.NUM_CLASSES,
            "class_names": Config.CLASS_NAMES,
            "learning_rate": Config.LEARNING_RATE,
            "momentum": Config.MOMENTUM,
            "batch_size": Config.BATCH_SIZE,
            "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
            "iou_thresholds": Config.IOU_THRESHOLDS,
            "image_size": Config.IMAGE_SIZE
        })
        
        self.log_data["system_info"] = system_info
    
    def log_data_info(self, df_image_info, num_train_images=None, num_val_images=None):
        """Log dataset information"""
        data_info = {
            "total_annotations": int(len(df_image_info)),
            "unique_images": int(df_image_info['filename'].nunique()),
            "class_distribution": {str(k): int(v) for k, v in df_image_info['class'].value_counts().to_dict().items()},
            "duplicate_rows": int(df_image_info.duplicated().sum()),
            "duplicate_filenames": int(df_image_info['filename'].duplicated().sum()),
            "missing_values": {str(k): int(v) for k, v in df_image_info.isnull().sum().to_dict().items()}
        }
        
        if num_train_images:
            data_info["num_train_images"] = int(num_train_images)
        if num_val_images:
            data_info["num_val_images"] = int(num_val_images)
        
        self.log_data["data_info"] = data_info
    
    def log_model_info(self, model_action, model_path=None):
        """Log model information"""
        model_info = {
            "action": model_action,  # "loaded" or "trained"
            "model_path": model_path,
            "model_architecture": "Faster R-CNN ResNet50 FPN"
        }
        
        self.log_data["model_info"] = model_info
    
    def start_training_timer(self):
        """Start training timer"""
        self.training_start_time = time.time()
    
    def log_epoch_metrics(self, epoch, train_loss, val_loss):
        """Log metrics for a single epoch"""
        if "epochs" not in self.log_data["training_metrics"]:
            self.log_data["training_metrics"]["epochs"] = []
        
        epoch_data = {
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss),
            "validation_loss": float(val_loss),
            "timestamp": datetime.now().isoformat()
        }
        
        self.log_data["training_metrics"]["epochs"].append(epoch_data)
    
    def log_training_completion(self):
        """Log training completion and calculate total time"""
        training_time = time.time() - self.training_start_time
        
        self.log_data["training_metrics"]["total_training_time_seconds"] = float(training_time)
        self.log_data["training_metrics"]["total_training_time_formatted"] = self._format_time(training_time)
        self.log_data["training_metrics"]["training_completed"] = datetime.now().isoformat()
    
    def log_evaluation_metrics(self, metrics):
        """Log evaluation metrics"""
        if metrics and 'iou_scores' in metrics and metrics['iou_scores']:
            eval_metrics = {
                "average_iou": float(torch.tensor(metrics['iou_scores']).mean()),
                "num_predictions": int(len(metrics['iou_scores'])),
                "iou_thresholds": {}
            }
            
            # Calculate metrics for each IoU threshold
            for threshold in Config.IOU_THRESHOLDS:
                above_threshold = sum(1 for iou in metrics['iou_scores'] if iou >= threshold)
                precision = above_threshold / len(metrics['iou_scores']) if metrics['iou_scores'] else 0
                eval_metrics["iou_thresholds"][f"threshold_{threshold}"] = {
                    "precision": float(precision),
                    "predictions_above_threshold": int(above_threshold)
                }
            
            self.log_data["validation_metrics"] = eval_metrics
        else:
            self.log_data["validation_metrics"] = {
                "message": "No predictions above confidence threshold found"
            }
    
    def log_inference_results(self, validation_results=None, webcam_used=False):
        """Log inference results"""
        inference_data = {
            "webcam_detection_used": bool(webcam_used),
            "validation_inference_completed": validation_results is not None
        }
        
        if validation_results:
            inference_data.update({
                "num_validation_images_processed": int(len(validation_results)),
                "validation_results_summary": {
                    "successful_predictions": int(len([r for r in validation_results if r is not None]))
                }
            })
        
        self.log_data["inference_results"] = inference_data
    
    def log_evaluation_results(self, evaluation_results):
        """Log ground truth evaluation results"""
        if evaluation_results:
            eval_data = {
                "ground_truth_evaluation": {
                    "total_predictions": evaluation_results.get('total_predictions', 0),
                    "correct_predictions": evaluation_results.get('correct_predictions', 0),
                    "accuracy": evaluation_results.get('accuracy', 0.0),
                    "accuracy_percentage": evaluation_results.get('accuracy', 0.0) * 100,
                    "class_statistics": {}
                }
            }
            
            # Process class statistics
            class_stats = evaluation_results.get('class_stats', {})
            for class_name, stats in class_stats.items():
                tp, fp, fn = stats.get('tp', 0), stats.get('fp', 0), stats.get('fn', 0)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                eval_data["ground_truth_evaluation"]["class_statistics"][class_name] = {
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                }
            
            self.log_data["ground_truth_evaluation"] = eval_data["ground_truth_evaluation"]
        else:
            self.log_data["ground_truth_evaluation"] = {
                "message": "Ground truth evaluation was not performed"
            }
    
    def save_logs(self):
        """Save all logs to files"""
        # Calculate total execution time
        total_time = time.time() - self.start_time
        self.log_data["timing"]["total_execution_time_seconds"] = float(total_time)
        self.log_data["timing"]["total_execution_time_formatted"] = self._format_time(total_time)
        self.log_data["execution_info"]["end_time"] = datetime.now().isoformat()
        
        # Convert all data to JSON-serializable format
        serializable_data = self._convert_to_serializable(self.log_data)
        
        # Save main log file as JSON
        log_file_path = os.path.join(self.log_folder, "execution_log.json")
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        # Save human-readable summary
        summary_path = os.path.join(self.log_folder, "summary.txt")
        self._save_human_readable_summary(summary_path)
        
        # Save training metrics as CSV if training occurred
        if self.log_data["training_metrics"].get("epochs"):
            self._save_training_csv()
        
        print(f"📝 Logs saved to: {self.log_folder}")
        print(f"   - execution_log.json (complete data)")
        print(f"   - summary.txt (human-readable)")
        if self.log_data["training_metrics"].get("epochs"):
            print(f"   - training_metrics.csv (epoch details)")
    
    def _save_human_readable_summary(self, summary_path):
        """Save a human-readable summary of the execution"""
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("ROCK PAPER SCISSORS DETECTION - EXECUTION SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            # Execution info
            f.write("EXECUTION INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Start Time: {self.log_data['execution_info']['start_time']}\n")
            f.write(f"End Time: {self.log_data['execution_info']['end_time']}\n")
            f.write(f"Execution Type: {self.log_data['execution_info']['execution_type']}\n")
            if self.log_data['execution_info']['num_epochs']:
                f.write(f"Number of Epochs: {self.log_data['execution_info']['num_epochs']}\n")
            f.write(f"Total Execution Time: {self.log_data['timing']['total_execution_time_formatted']}\n\n")
            
            # System info
            f.write("SYSTEM INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Device: {self.log_data['system_info']['device_type'].upper()}\n")
            f.write(f"PyTorch Version: {self.log_data['system_info']['pytorch_version']}\n")
            f.write(f"CUDA Available: {self.log_data['system_info']['cuda_available']}\n")
            if self.log_data['system_info']['cuda_available']:
                f.write(f"GPU: {self.log_data['system_info']['gpu_name']}\n")
                f.write(f"CUDA Version: {self.log_data['system_info']['cuda_version']}\n")
            f.write("\n")
            
            # Data info
            if self.log_data['data_info']:
                f.write("DATASET INFORMATION:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Annotations: {self.log_data['data_info']['total_annotations']}\n")
                f.write(f"Unique Images: {self.log_data['data_info']['unique_images']}\n")
                f.write("Class Distribution:\n")
                for class_name, count in self.log_data['data_info']['class_distribution'].items():
                    f.write(f"  {class_name}: {count}\n")
                f.write("\n")
            
            # Training metrics
            if self.log_data['training_metrics'].get('epochs'):
                f.write("TRAINING RESULTS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Training Time: {self.log_data['training_metrics']['total_training_time_formatted']}\n")
                f.write(f"Number of Epochs: {len(self.log_data['training_metrics']['epochs'])}\n")
                
                # Show last epoch metrics
                last_epoch = self.log_data['training_metrics']['epochs'][-1]
                f.write(f"Final Training Loss: {last_epoch['train_loss']:.4f}\n")
                f.write(f"Final Validation Loss: {last_epoch['validation_loss']:.4f}\n\n")
            
            # Validation metrics
            if self.log_data['validation_metrics']:
                f.write("VALIDATION RESULTS:\n")
                f.write("-" * 30 + "\n")
                if 'average_iou' in self.log_data['validation_metrics']:
                    f.write(f"Average IoU: {self.log_data['validation_metrics']['average_iou']:.4f}\n")
                    f.write(f"Number of Predictions: {self.log_data['validation_metrics']['num_predictions']}\n")
                    
                    for threshold_key, metrics in self.log_data['validation_metrics']['iou_thresholds'].items():
                        threshold = threshold_key.replace('threshold_', '')
                        f.write(f"Precision @ IoU {threshold}: {metrics['precision']:.4f}\n")
                else:
                    f.write("No predictions above confidence threshold\n")
                f.write("\n")
            
            # Inference results
            if self.log_data['inference_results']:
                f.write("INFERENCE RESULTS:\n")
                f.write("-" * 30 + "\n")
                if self.log_data['inference_results']['validation_inference_completed']:
                    f.write(f"Validation Images Processed: {self.log_data['inference_results']['num_validation_images_processed']}\n")
                if self.log_data['inference_results']['webcam_detection_used']:
                    f.write("Webcam Detection: Executed\n")
                f.write("\n")
    
    def _save_training_csv(self):
        """Save training metrics as CSV"""
        try:
            import pandas as pd
            epochs_data = self.log_data['training_metrics']['epochs']
            df = pd.DataFrame(epochs_data)
            csv_path = os.path.join(self.log_folder, "training_metrics.csv")
            df.to_csv(csv_path, index=False)
        except ImportError:
            # If pandas is not available, save as simple CSV
            csv_path = os.path.join(self.log_folder, "training_metrics.csv")
            with open(csv_path, 'w') as f:
                f.write("epoch,train_loss,validation_loss,timestamp\n")
                for epoch_data in self.log_data['training_metrics']['epochs']:
                    f.write(f"{epoch_data['epoch']},{epoch_data['train_loss']},{epoch_data['validation_loss']},{epoch_data['timestamp']}\n")
    
    def _format_time(self, seconds):
        """Format time in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def get_log_folder(self):
        """Get the path to the current log folder"""
        return self.log_folder
