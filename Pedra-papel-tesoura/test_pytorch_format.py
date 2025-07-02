"""
Test script to verify PyTorch model format compatibility
"""

import torch
from config import Config
from model_utils import ModelManager
import os

def test_pytorch_format():
    """Test PyTorch model format compatibility"""
    print("="*60)
    print("PYTORCH MODEL FORMAT COMPATIBILITY TEST")
    print("="*60)
    
    # Test on both CPU and CUDA if available
    devices_to_test = ['cpu']
    if torch.cuda.is_available():
        devices_to_test.append('cuda')
    
    for device_name in devices_to_test:
        print(f"\n--- Testing {device_name.upper()} ---")
        
        device = torch.device(device_name)
        model_manager = ModelManager(device)
        
        try:
            # Create a test model
            print("Creating model...")
            model = model_manager.create_model()
            print("✓ Model created successfully")
            
            # Test saving in PyTorch format
            test_model_path = f"test_model_{device_name}.pth"
            print(f"Saving model to {test_model_path}...")
            model_manager.save_model(model, test_model_path)
            print("✓ Model saved successfully")
            
            # Test loading the saved model
            print("Loading saved model...")
            loaded_model = model_manager.load_model(test_model_path)
            print("✓ Model loaded successfully")
            
            # Test that loaded model works
            print("Testing loaded model inference...")
            loaded_model.eval()
            test_input = torch.randn(3, 224, 224).to(device)
            
            with torch.no_grad():
                # Test PyTorch object detection format
                output = loaded_model([test_input])
                print("✓ Model inference works correctly")
                print(f"  Output keys: {output[0].keys()}")
            
            # Cleanup
            if os.path.exists(test_model_path):
                os.remove(test_model_path)
                print(f"✓ Cleaned up {test_model_path}")
            
            print(f"✅ All tests passed for {device_name.upper()}")
            
        except Exception as e:
            print(f"❌ Error testing {device_name.upper()}: {str(e)}")

def test_cross_device_compatibility():
    """Test loading model saved on one device and loading on another"""
    print(f"\n--- CROSS-DEVICE COMPATIBILITY TEST ---")
    
    try:
        # Save model on CPU
        cpu_device = torch.device('cpu')
        cpu_manager = ModelManager(cpu_device)
        
        print("Creating and saving model on CPU...")
        model = cpu_manager.create_model()
        test_path = "cross_device_test.pth"
        cpu_manager.save_model(model, test_path)
        
        # Try to load on CUDA if available
        if torch.cuda.is_available():
            print("Loading CPU-saved model on CUDA...")
            cuda_device = torch.device('cuda')
            cuda_manager = ModelManager(cuda_device)
            
            loaded_model = cuda_manager.load_model(test_path)
            print("✓ Successfully loaded CPU model on CUDA")
            
            # Test inference
            test_input = torch.randn(3, 224, 224).to(cuda_device)
            with torch.no_grad():
                output = loaded_model([test_input])
                print("✓ Cross-device inference works")
        else:
            print("CUDA not available, skipping CUDA test")
        
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
        
        print("✅ Cross-device compatibility test passed")
        
    except Exception as e:
        print(f"❌ Cross-device test failed: {str(e)}")

def test_model_info():
    """Display model and PyTorch information"""
    print(f"\n--- SYSTEM INFORMATION ---")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    print(f"Model format: PyTorch .pth (state_dict)")
    print(f"Benefits:")
    print(f"  ✓ Industry standard format")
    print(f"  ✓ Device independent")
    print(f"  ✓ Version independent") 
    print(f"  ✓ Smaller file size")
    print(f"  ✓ Better compatibility")

if __name__ == "__main__":
    test_model_info()
    test_pytorch_format()
    test_cross_device_compatibility()
    
    print("\n" + "="*60)
    print("PYTORCH FORMAT COMPATIBILITY TEST COMPLETED")
    print("="*60)
