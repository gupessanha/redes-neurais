"""
Training utilities for Rock Paper Scissors detection - GPU optimized
"""
import torch
import numpy as np
import time
import gc
from collections import defaultdict
from config import Config

class Trainer:
    """GPU-optimized training utilities for the model"""
    
    def __init__(self, model, optimizer, device, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        
        # Configurar otimizações GPU
        self._setup_gpu_optimizations()
        
        # Métricas de performance
        self.batch_times = []
        self.gpu_memory_usage = []
        
    def _setup_gpu_optimizations(self):
        """Configurar otimizações específicas da GPU"""
        if self.device.type == 'cuda':
            # Mixed precision training
            if Config.MIXED_PRECISION and hasattr(torch.cuda, 'amp'):
                self.use_amp = True
                self.scaler = torch.cuda.amp.GradScaler()
                print("⚡ Mixed Precision (AMP) enabled")
            else:
                self.use_amp = False
                
            # Configurar GPU
            Config.setup_gpu_optimizations()
        else:
            self.use_amp = False
    
    def train_one_epoch(self, data_loader, epoch=1):
        """Train model for one epoch with GPU optimizations"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()
        
        for batch_idx, (images, targets) in enumerate(data_loader):
            batch_start_time = time.time()
            
            # Transferência otimizada para GPU
            if self.device.type == 'cuda':
                images = [image.to(self.device, non_blocking=True) for image in images]
                targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]
            else:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Zero gradients mais eficiente
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass com mixed precision se disponível
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                # Backward pass com gradient scaling
                self.scaler.scale(losses).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Verificar loss finito
                if not torch.isfinite(losses):
                    print(f"⚠️ Non-finite loss detected: {losses}")
                    continue
                
                losses.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Monitoramento de performance GPU
            if self.device.type == 'cuda':
                current_memory = torch.cuda.memory_allocated() / 1024**3
                self.gpu_memory_usage.append(current_memory)
                
                # Limpeza periódica de cache
                if batch_idx % Config.EMPTY_CACHE_FREQUENCY == 0:
                    torch.cuda.empty_cache()
            
            total_loss += losses.detach().cpu().item()
            num_batches += 1
            
            # Log de progresso
            batch_time = time.time() - batch_start_time
            self.batch_times.append(batch_time)
            
            if batch_idx % 10 == 0:
                avg_batch_time = sum(self.batch_times[-10:]) / min(len(self.batch_times), 10)
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(data_loader)}: "
                      f"Loss={losses.item():.4f}, Time={avg_batch_time:.3f}s", end="")
                
                if self.device.type == 'cuda':
                    print(f", GPU Mem={current_memory:.1f}GB")
                else:
                    print()
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Estatísticas da época
        if self.device.type == 'cuda' and self.gpu_memory_usage:
            avg_gpu_memory = sum(self.gpu_memory_usage[-num_batches:]) / num_batches
            max_gpu_memory = max(self.gpu_memory_usage[-num_batches:])
            print(f"📊 Epoch {epoch}: Loss={avg_loss:.4f}, Time={epoch_time:.1f}s, "
                  f"Avg GPU={avg_gpu_memory:.1f}GB, Max GPU={max_gpu_memory:.1f}GB")
        
        return avg_loss
    
    def validate(self, data_loader):
        """Validate model with GPU optimizations"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in data_loader:
                # Transferência otimizada para GPU
                if self.device.type == 'cuda':
                    images = [image.to(self.device, non_blocking=True) for image in images]
                    targets = [{k: v.to(self.device, non_blocking=True) for k, v in t.items()} for t in targets]
                else:
                    images = [image.to(self.device) for image in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Switch to training mode temporarily for loss calculation
                self.model.train()
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        loss_dict = self.model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                else:
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                self.model.eval()
                
                total_loss += losses.cpu().item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train_model(self, train_loader, valid_loader):
        """Complete training loop with GPU optimizations"""
        print("\n🚀 Starting optimized training...")
        print(f"📱 Training on device: {self.device}")
        print(f"🔢 Number of epochs: {Config.NUM_EPOCHS}")
        print(f"📊 Batch size: {Config.BATCH_SIZE}")
        print(f"🎯 Learning rate: {Config.LEARNING_RATE}")
        print(f"⚡ Mixed precision: {self.use_amp}")
        print("-" * 50)
        
        training_start_time = time.time()
        best_val_loss = float('inf')
        
        for epoch in range(Config.NUM_EPOCHS):
            epoch_start = time.time()
            
            # Treinamento
            train_loss = self.train_one_epoch(train_loader, epoch + 1)
            
            # Validação
            val_loss = self.validate(valid_loader)
            
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"✅ Epoch {epoch+1}/{Config.NUM_EPOCHS} completed in {epoch_time:.1f}s")
            print(f"📈 Training Loss: {train_loss:.4f}")
            print(f"📉 Validation Loss: {val_loss:.4f}")
            print(f"🎯 Learning Rate: {current_lr:.6f}")
            
            # Verificar melhor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"🏆 New best validation loss: {val_loss:.4f}")
            
            # Log epoch metrics if logger is available
            if self.logger:
                self.logger.log_epoch_metrics(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=current_lr,
                    epoch_time=epoch_time
                )
            
            # Limpeza de memória GPU entre épocas
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            print("-" * 50)
        
        total_training_time = time.time() - training_start_time
        print(f"🎉 Training completed in {total_training_time/60:.1f} minutes!")
        
        # Estatísticas finais de GPU
        if self.device.type == 'cuda' and self.gpu_memory_usage:
            avg_gpu_usage = sum(self.gpu_memory_usage) / len(self.gpu_memory_usage)
            max_gpu_usage = max(self.gpu_memory_usage)
            print(f"📊 GPU Usage Stats - Average: {avg_gpu_usage:.1f}GB, Peak: {max_gpu_usage:.1f}GB")
        
        return total_training_time

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
