"""
Data handling utilities for Rock Paper Scissors detection
"""
import pandas as pd
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from config import Config

class RPSDataset(Dataset):
    """Custom Dataset for Rock Paper Scissors detection"""
    
    def __init__(self, image_tensors, targets):
        self.images = image_tensors
        self.targets = targets
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]

class DataLoader:
    """Data loading and processing utilities"""
    
    def __init__(self):
        self.transform = T.Compose([T.ToTensor()])
    
    def load_annotations(self, csv_path):
        """Load and examine the dataset annotations"""
        print("Loading dataset...")
        df_image_info = pd.read_csv(csv_path)
        print(f"Dataset shape: {df_image_info.shape}")
        print(df_image_info.head())
        return df_image_info
    
    def quality_check(self, df_image_info):
        """Perform data quality checks"""
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
    
    def visualize_class_distribution(self, df_image_info):
        """Visualize the distribution of classes"""
        print("\nVisualizing class distribution...")
        plt.style.use('_mpl-gallery')
        class_counts = df_image_info['class'].value_counts()
        
        plt.figure(figsize=(8, 6))
        plt.bar(class_counts.index, class_counts.values, alpha=0.7)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Distribution of Rock, Paper, Scissors Classes')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def load_training_data(self, df_image_info, images_path):
        """Load and process training images"""
        print("\nLoading and processing training images...")
        
        image_tensors = []
        targets = []
        
        for i in range(len(df_image_info)):
            image_path = os.path.join(images_path, df_image_info.loc[i, 'filename'])
            img = cv2.imread(image_path)
            
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                x1 = df_image_info.loc[i, 'xmin']
                y1 = df_image_info.loc[i, 'ymin']
                x2 = df_image_info.loc[i, 'xmax']
                y2 = df_image_info.loc[i, 'ymax']
                
                # Map class names to numeric labels
                class_name = df_image_info.loc[i, 'class']
                if class_name in ['rock', 'Rock']:
                    label = 1
                elif class_name in ['paper', 'Paper']:
                    label = 2
                elif class_name in ['scissors', 'Scissors']:
                    label = 3
                else:
                    label = 1  # Default to rock
                
                boxes = [[x1, y1, x2, y2]]
                labels = [label]
                
                target = {
                    "boxes": torch.tensor(boxes, dtype=torch.float32),
                    "labels": torch.tensor(labels, dtype=torch.int64)
                }
                
                img_tensor = self.transform(img)
                image_tensors.append(img_tensor)
                targets.append(target)
        
        print(f"Loaded {len(image_tensors)} images successfully")
        return image_tensors, targets
    
    def load_validation_images(self, validation_path):
        """Load validation images for testing"""
        print(f"\nLoading validation images from {validation_path}...")
        
        # Support common image formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        validation_images = []
        
        for extension in image_extensions:
            validation_images.extend(glob.glob(os.path.join(validation_path, extension)))
        
        print(f"Found {len(validation_images)} validation images")
        return validation_images
    
    def create_data_loaders(self, image_tensors, targets):
        """Create optimized training and validation data loaders for GPU"""
        print("🔄 Creating optimized data loaders...")
        dataset = RPSDataset(image_tensors, targets)
        
        # Split data into training and validation
        indices = torch.randperm(len(dataset)).tolist()
        train_dataset = torch.utils.data.Subset(dataset, indices[:-Config.VALIDATION_SPLIT])
        valid_dataset = torch.utils.data.Subset(dataset, indices[-Config.VALIDATION_SPLIT:])
        
        # Configurar workers baseado na disponibilidade
        num_workers = Config.NUM_WORKERS if Config.NUM_WORKERS > 0 else 0
        
        # DataLoader de treinamento otimizado
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True, 
            collate_fn=lambda x: tuple(zip(*x)),
            num_workers=num_workers,
            pin_memory=Config.PIN_MEMORY and torch.cuda.is_available(),
            persistent_workers=Config.PERSISTENT_WORKERS and num_workers > 0,
            prefetch_factor=Config.PREFETCH_FACTOR if num_workers > 0 else 2,
            drop_last=True  # Evita batches pequenos que podem causar problemas
        )
        
        # DataLoader de validação otimizado
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            collate_fn=lambda x: tuple(zip(*x)),
            num_workers=num_workers,
            pin_memory=Config.PIN_MEMORY and torch.cuda.is_available(),
            persistent_workers=Config.PERSISTENT_WORKERS and num_workers > 0,
            prefetch_factor=Config.PREFETCH_FACTOR if num_workers > 0 else 2
        )
        
        print(f"📊 Training batches: {len(train_loader)} (batch_size={Config.BATCH_SIZE})")
        print(f"📊 Validation batches: {len(valid_loader)}")
        print(f"🔧 Workers: {num_workers}, Pin Memory: {Config.PIN_MEMORY}")
        print(f"🚀 Persistent Workers: {Config.PERSISTENT_WORKERS}, Prefetch: {Config.PREFETCH_FACTOR}")
        
        return train_loader, valid_loader
    
    def get_transform(self, train=False):
        """Get image transforms for training or validation"""
        transform_list = [
            T.Resize(Config.IMAGE_SIZE),
            T.ToTensor(),
        ]
        if train:
            transform_list.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transform_list)
