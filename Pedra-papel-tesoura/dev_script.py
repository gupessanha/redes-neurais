"""
Main script for Rock Paper Scissors object detection
Modularized version with CPU/CUDA support and validation data integration
"""
import os
from config import Config
from data_utils import DataLoader
from model_utils import ModelManager
from training_utils import Trainer, Evaluator
from inference_utils import InferenceEngine, WebcamDetection
from logging_utils import ExecutionLogger

def main():
    """Main function for Rock Paper Scissors object detection"""
    
    print("="*60)
    print("ROCK PAPER SCISSORS DETECTION SYSTEM")
    print("="*60)
    
    # Initialize logger - will be updated once we know if training or inference
    logger = None
    
    # Check system requirements and paths
    print("\n1. System Check")
    print("-" * 30)
    
    # Detect device (CPU/CUDA)
    device = Config.get_device()
    
    # Check if required paths exist
    if not Config.check_paths():
        print("Please ensure all required directories exist before continuing.")
        return
    
    # Initialize components
    data_loader = DataLoader()
    model_manager = ModelManager(device)
    
    print("\n2. Data Loading and Analysis")
    print("-" * 30)
    
    # Load and analyze dataset
    df_image_info = data_loader.load_annotations(Config.TRAIN_DATA_PATH)
    data_loader.quality_check(df_image_info)
    
    print("\n3. Model Setup")
    print("-" * 30)
    
    # Model setup - Choose to load existing model or train new one
    load_existing, model_path = model_manager.get_model_choice(Config.MODEL_FILENAME)
    
    if load_existing:
        # Initialize logger for inference
        logger = ExecutionLogger(num_epochs=None)
        logger.log_data_info(df_image_info)
        logger.log_model_info("loaded", model_path)
        
        # Load existing model
        model = model_manager.load_model(model_path)
    else:
        # Initialize logger for training
        logger = ExecutionLogger(num_epochs=Config.NUM_EPOCHS)
        logger.log_data_info(df_image_info)
        logger.log_model_info("trained", model_path)
        
        # Train new model
        model = train_new_model(data_loader, df_image_info, model_manager, device, model_path, logger)
    
    # Apply device-specific optimizations
    model = model_manager.optimize_for_device(model)
    
    print("\n4. Choose Operation Mode")
    print("-" * 30)
    print("1. Run inference on validation images")
    print("2. Start webcam detection") 
    print("3. Both")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Please enter 1, 2, or 3.")
    
    validation_results = None
    webcam_used = False
    
    if choice in ['1', '3']:
        # Run validation inference
        print("\n5. Validation Inference")
        print("-" * 30)
        validation_results = run_validation_inference(model, device, logger)
    
    if choice in ['2', '3']:
        # Start webcam detection
        print("\n6. Webcam Detection")
        print("-" * 30)
        webcam_used = run_webcam_detection(model, device)
    
    # Log inference results and save all logs
    logger.log_inference_results(validation_results, webcam_used)
    logger.save_logs()
    
    print("\nProgram completed successfully!")

def train_new_model(data_loader, df_image_info, model_manager, device, model_path, logger):
    """Train and save a new model"""
    print("\nTraining new model...")
    
    # Start training timer
    logger.start_training_timer()
    
    # Load training data
    image_tensors, targets = data_loader.load_training_data(df_image_info, Config.TRAIN_IMAGES_PATH)
    
    # Create data loaders
    train_loader, valid_loader = data_loader.create_data_loaders(image_tensors, targets)
    
    # Create model
    model = model_manager.create_model()
    
    # Setup optimizer
    optimizer = model_manager.setup_optimizer(model)
    
    # Create trainer and train
    trainer = Trainer(model, optimizer, device, logger)
    trainer.train_model(train_loader, valid_loader)
    
    # Log training completion
    logger.log_training_completion()
    
    # Save model using PyTorch format in the log folder
    log_folder = logger.get_log_folder()
    model = model_manager.save_model(model, model_path, log_folder)
    
    # Evaluate model
    evaluator = Evaluator(device)
    metrics = evaluator.evaluate_model(model, valid_loader)
    
    # Log evaluation metrics
    logger.log_evaluation_metrics(metrics)
    
    return model

def run_validation_inference(model, device, logger):
    """Run inference on validation images and evaluate against ground truth"""
    data_loader = DataLoader()
    validation_images = data_loader.load_validation_images(Config.VALIDATION_IMAGES_PATH)
    
    if not validation_images:
        print("No validation images found!")
        return None
    
    # Run basic inference
    inference_engine = InferenceEngine(model, device)
    results = inference_engine.predict_validation_images(validation_images)
    
    print(f"\nValidation inference completed on {len(results)} images.")
    
    # Evaluate against ground truth if annotations file exists
    test_annotations_path = os.path.join(Config.VALIDATION_IMAGES_PATH, '_annotations.csv')
    if os.path.exists(test_annotations_path):
        print(f"\nFound ground truth annotations at {test_annotations_path}")
        print("Running evaluation against ground truth...")
        
        from inference_utils import ModelEvaluator
        evaluator = ModelEvaluator(model, device)
        evaluation_results = evaluator.evaluate_model(Config.VALIDATION_IMAGES_PATH, test_annotations_path)
        
        if evaluation_results:
            # Log evaluation results
            logger.log_evaluation_results(evaluation_results)
        
        return {'inference_results': results, 'evaluation_results': evaluation_results}
    else:
        print(f"\nNo ground truth annotations found at {test_annotations_path}")
        print("Skipping evaluation against ground truth.")
        return {'inference_results': results, 'evaluation_results': None}

def run_webcam_detection(model, device):
    """Run real-time webcam detection"""
    webcam_detector = WebcamDetection(model, device)
    
    # Test webcam access first
    if not webcam_detector.test_webcam_access():
        print("Webcam not accessible. Skipping webcam detection.")
        return False
    
    webcam_detector.run_webcam_detection()
    return True


if __name__ == "__main__":
    main()

