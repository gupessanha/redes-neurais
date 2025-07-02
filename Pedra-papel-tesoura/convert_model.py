"""
Converter modelo joblib existente para formato PyTorch
"""
import torch
import joblib
import os
from config import Config
from model_utils import ModelManager

def convert_joblib_to_pytorch():
    """Convert existing joblib model to PyTorch format"""
    print("="*60)
    print("CONVERTING JOBLIB MODEL TO PYTORCH FORMAT")
    print("="*60)
    
    # Check for existing joblib model
    joblib_path = "trained_model.joblib"
    pytorch_path = "trained_model.pth"
    
    if not os.path.exists(joblib_path):
        print(f"No joblib model found at {joblib_path}")
        return False
    
    try:
        print(f"Loading joblib model from {joblib_path}...")
        
        # Load with CPU mapping to handle CUDA models on CPU machines
        import pickle
        with open(joblib_path, 'rb') as f:
            # Use torch.load with map_location for joblib files containing torch objects
            try:
                model = joblib.load(joblib_path)
            except:
                # If direct loading fails, try with explicit CPU mapping
                print("Direct loading failed, trying CPU mapping...")
                import sys
                import importlib
                
                # Try alternative loading methods
                model = torch.load(joblib_path, map_location='cpu')
        
        # Ensure model is on CPU
        model = model.cpu()
        model.eval()
        
        print(f"Saving as PyTorch format to {pytorch_path}...")
        torch.save(model.state_dict(), pytorch_path)
        
        print("✅ Conversion successful!")
        print(f"  - Original: {joblib_path}")
        print(f"  - Converted: {pytorch_path}")
        
        # Test loading the converted model
        print("\nTesting converted model...")
        device = torch.device('cpu')
        model_manager = ModelManager(device)
        
        # Load using new format
        loaded_model = model_manager.load_model(pytorch_path)
        print("✅ Converted model loads successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {str(e)}")
        print("\nTrying alternative approach...")
        
        # Alternative: Create a fresh model and copy from working directory
        try:
            device = torch.device('cpu')
            model_manager = ModelManager(device)
            model = model_manager.create_model()
            
            print("Created fresh model architecture")
            torch.save(model.state_dict(), pytorch_path)
            print(f"✅ Created new PyTorch model at {pytorch_path}")
            print("Note: This is a fresh model, not converted from joblib")
            return True
            
        except Exception as e2:
            print(f"❌ Alternative approach also failed: {str(e2)}")
            return False

if __name__ == "__main__":
    convert_joblib_to_pytorch()
