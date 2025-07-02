"""
Model utilities for Rock Paper Scissors detection
"""
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
from config import Config

class ModelManager:
    """Model management utilities"""
    
    def __init__(self, device):
        self.device = device
    
    def create_model(self):
        """Create and configure the Faster R-CNN model"""
        print("\nSetting up model...")
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        
        # Replace the classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, Config.NUM_CLASSES)
        
        return model.to(self.device)
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        print(f"Loading model from {model_path}...")
        
        try:
            # Create a fresh model architecture
            model = self.create_model()
            
            # Load the state dict with device mapping for compatibility
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            
            # Ensure model is on correct device and in eval mode
            model = model.to(self.device)
            model.eval()
            
            print("Model loaded successfully!")
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Please check if the model file exists and is compatible.")
            raise
    
    def save_model(self, model, model_path, log_folder=None):
        """Save the trained model using PyTorch standard format"""
        
        # If log_folder is provided and Config allows, save model in the log folder
        if log_folder and Config.SAVE_MODELS_IN_LOGS:
            model_filename = os.path.basename(model_path)
            new_model_path = os.path.join(log_folder, model_filename)
            
            # Clean up old models from root directory if enabled
            if Config.CLEANUP_OLD_MODELS:
                self._cleanup_old_models(model_path)
            
            model_path = new_model_path
        
        print(f"\nSaving model to {model_path}...")
        
        try:
            # Ensure the model path has .pth extension
            if not model_path.endswith('.pth'):
                model_path = model_path.replace('.joblib', '.pth')
            
            # Move model to CPU before saving for device-independent compatibility
            model_cpu = model.cpu()
            
            # Save only the state_dict (recommended PyTorch practice)
            torch.save(model_cpu.state_dict(), model_path)
            
            # Move model back to original device
            model = model.to(self.device)
            
            print(f"Model saved successfully as {model_path}")
            print("✓ Compatible with any PyTorch installation")
            print("✓ Device-independent format")
            if log_folder and Config.SAVE_MODELS_IN_LOGS:
                print("✓ Saved in log folder with execution data")
            
            return model
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            # Ensure model is moved back to original device even if save failed
            model = model.to(self.device)
            raise
    
    def _cleanup_old_models(self, original_model_path):
        """Remove old model files from root directory"""
        try:
            # Get base name without extension for cleanup
            base_name = os.path.splitext(os.path.basename(original_model_path))[0]
            
            # List of model file patterns to clean up
            patterns_to_cleanup = [
                f"{base_name}.pth",
                f"{base_name}.joblib",
                "trained_model.pth",
                "trained_model.joblib"
            ]
            
            cleaned_files = []
            for pattern in patterns_to_cleanup:
                if os.path.exists(pattern):
                    os.remove(pattern)
                    cleaned_files.append(pattern)
            
            if cleaned_files:
                print(f"🧹 Cleaned up old model files: {', '.join(cleaned_files)}")
                
        except Exception as e:
            print(f"Warning: Could not clean up old model files: {str(e)}")
            # Don't raise exception as this is not critical
    
    def model_exists(self, model_path):
        """Check if model file exists"""
        # Check for both .pth and .joblib formats for backward compatibility
        pth_path = model_path.replace('.joblib', '.pth')
        return os.path.exists(pth_path) or os.path.exists(model_path)
    
    def get_model_choice(self, model_path):
        """Ask user whether to load existing model or train new one"""
        # Check for PyTorch format first, then joblib for backward compatibility
        pth_path = model_path.replace('.joblib', '.pth')
        
        if os.path.exists(pth_path):
            print(f"\nFound existing PyTorch model at {pth_path}")
            model_path = pth_path  # Use PyTorch format
        elif os.path.exists(model_path):
            print(f"\nFound existing model at {model_path}")
            print("Note: Consider retraining to save in PyTorch format (.pth)")
        else:
            print(f"\nNo existing model found")
            print("Will train a new model...")
            return False, model_path.replace('.joblib', '.pth')
        
        while True:
            choice = input("Do you want to load the existing model? (y/n): ").lower().strip()
            if choice in ['y', 'yes']:
                return True, model_path
            elif choice in ['n', 'no']:
                return False, pth_path
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    
    def setup_optimizer(self, model):
        """Setup optimizer for training"""
        return torch.optim.SGD(
            model.parameters(), 
            lr=Config.LEARNING_RATE, 
            momentum=Config.MOMENTUM
        )
    
    def print_device_info(self):
        """Print device information"""
        print(f"\nDevice: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        else:
            print("Running on CPU")
    
    def optimize_for_device(self, model):
        """Apply device-specific optimizations"""
        if self.device.type == 'cuda':
            # Enable mixed precision training if available
            model = model.to(self.device)
            # You could add torch.compile() here for PyTorch 2.0+
            print("Model optimized for CUDA")
        else:
            # CPU optimizations
            torch.set_num_threads(torch.get_num_threads())
            print(f"Model optimized for CPU with {torch.get_num_threads()} threads")
        
        return model
