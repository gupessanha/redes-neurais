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
    
    # Training parameters - Otimizado para GPU
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    BATCH_SIZE = 16  # Aumentado para melhor uso da GPU
    VALIDATION_SPLIT = 50  # Number of images for validation
    
    # DataLoader optimizations - Elimina gargalos de GPU
    NUM_WORKERS = 4  # Workers paralelos para carregamento
    PIN_MEMORY = True  # Acelera transferência CPU→GPU
    PREFETCH_FACTOR = 2  # Pre-carrega batches
    PERSISTENT_WORKERS = True  # Mantém workers vivos entre épocas
    
    # GPU optimizations
    MIXED_PRECISION = True  # Usar AMP para economia de memória
    GRADIENT_ACCUMULATION_STEPS = 1  # Para simular batch sizes maiores se necessário
    EMPTY_CACHE_FREQUENCY = 20  # Limpar cache GPU a cada N batches
    
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
    
    @staticmethod
    def setup_gpu_optimizations():
        """Configurar otimizações específicas da GPU"""
        if torch.cuda.is_available():
            # Habilitar cuDNN benchmark para otimizar convoluções
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Limpar cache inicial
            torch.cuda.empty_cache()
            
            # Informações da GPU
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory / 1024**3
            
            print(f"🚀 GPU Optimizations enabled:")
            print(f"   - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
            print(f"   - Total GPU Memory: {total_memory:.1f} GB")
            print(f"   - Mixed Precision: {Config.MIXED_PRECISION}")
            
            return True
        else:
            print("⚠️ GPU not available - running on CPU")
            return False
    
    @staticmethod
    def get_optimal_batch_size():
        """Determinar batch size ótimo baseado na GPU disponível"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory >= 20:
                return 32  # Para GPUs de 20GB+
            elif gpu_memory >= 12:
                return 24  # Para GPUs de 12GB+
            elif gpu_memory >= 8:
                return 16  # Para GPUs de 8GB+
            else:
                return 8   # Para GPUs menores
        return 4  # CPU fallback
