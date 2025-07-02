"""
Configuration module for Rock Paper Scissors detection
"""
import torch
import os

class Config:
    """Configuration class for the project"""
    
    # Paths
    TRAIN_DATA_PATH = './train/_annotations.csv'
    TRAIN_IMAGES_PATH = './train'  
    VALIDATION_IMAGES_PATH = './test'
    TEST_ANNOTATIONS_PATH = './test/_annotations.csv'
    MODEL_FILENAME = 'trained_model.pth'
    
    # Model parameters
    NUM_CLASSES = 4  # 3 classes (Rock, Paper, Scissors) + background
    CLASS_NAMES = ['Background', 'Rock', 'Paper', 'Scissors']
    
    # Training parameters
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    BATCH_SIZE = 4
    VALIDATION_SPLIT = 50  # Number of images for validation
    
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLDS = [0.5, 0.75]
    
    # Image parameters
    IMAGE_SIZE = (800, 800)
    
    # File management
    SAVE_MODELS_IN_LOGS = True  # Save models in log folders instead of root directory
    CLEANUP_OLD_MODELS = True   # Remove old models from root when saving in logs
    
    @staticmethod
    def get_device():
        """Detect and return the best available device (CUDA or CPU)"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"CUDA detected! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"PyTorch CUDA version: {torch.version.cuda}")
        else:
            device = torch.device("cpu")
            print("CUDA not available. Using CPU.")
        
        return device
    
    @staticmethod
    def check_paths():
        """Check if required paths exist"""
        paths_to_check = [
            Config.TRAIN_IMAGES_PATH,
            Config.VALIDATION_IMAGES_PATH,
            Config.TRAIN_DATA_PATH,
            Config.TEST_ANNOTATIONS_PATH
        ]
        
        missing_paths = []
        for path in paths_to_check:
            if not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_paths:
            print("Warning: The following paths are missing:")
            for path in missing_paths:
                print(f"  - {path}")
            return False
        return True
