"""
GPU Performance Monitor for Rock Paper Scissors Training
Monitora uso de GPU durante o treinamento para identificar gargalos
"""
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from config import Config
import psutil
import threading
import queue

class GPUMonitor:
    """Monitor de performance da GPU durante treinamento"""
    
    def __init__(self):
        self.gpu_stats = []
        self.cpu_stats = []
        self.memory_stats = []
        self.monitoring = False
        self.stats_queue = queue.Queue()
        
    def start_monitoring(self, interval=1.0):
        """Iniciar monitoramento em thread separada"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 GPU monitoring started")
    
    def stop_monitoring(self):
        """Parar monitoramento"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
        print("📊 GPU monitoring stopped")
    
    def _monitor_loop(self, interval):
        """Loop de monitoramento em background"""
        while self.monitoring:
            try:
                stats = self._collect_stats()
                self.stats_queue.put(stats)
                time.sleep(interval)
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                break
    
    def _collect_stats(self):
        """Coletar estatísticas do sistema"""
        stats = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / 1024**3,
        }
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            })
            
            # Utilização GPU (se disponível)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats['gpu_utilization'] = gpu_util.gpu
                stats['gpu_memory_util'] = gpu_util.memory
            except ImportError:
                stats['gpu_utilization'] = 0
                stats['gpu_memory_util'] = 0
        
        return stats
    
    def get_current_stats(self):
        """Obter estatísticas atuais"""
        # Processar queue de stats
        while not self.stats_queue.empty():
            try:
                stats = self.stats_queue.get_nowait()
                self.gpu_stats.append(stats)
            except queue.Empty:
                break
        
        if self.gpu_stats:
            return self.gpu_stats[-1]
        return self._collect_stats()
    
    def print_current_stats(self):
        """Imprimir estatísticas atuais"""
        stats = self.get_current_stats()
        
        print(f"\n📊 Current System Stats:")
        print(f"   CPU: {stats['cpu_percent']:.1f}%")
        print(f"   RAM: {stats['ram_used_gb']:.1f}GB ({stats['ram_percent']:.1f}%)")
        
        if torch.cuda.is_available():
            gpu_used = stats['gpu_memory_allocated']
            gpu_total = stats['gpu_memory_total']
            gpu_percent = (gpu_used / gpu_total) * 100
            
            print(f"   GPU Memory: {gpu_used:.1f}GB / {gpu_total:.1f}GB ({gpu_percent:.1f}%)")
            print(f"   GPU Utilization: {stats.get('gpu_utilization', 0):.1f}%")
    
    def diagnose_gpu_issues(self):
        """Diagnosticar possíveis problemas de GPU"""
        if not torch.cuda.is_available():
            print("❌ GPU not available")
            return
        
        stats = self.get_current_stats()
        issues = []
        
        # Verificar uso de memória GPU
        gpu_used = stats['gpu_memory_allocated']
        gpu_total = stats['gpu_memory_total']
        gpu_percent = (gpu_used / gpu_total) * 100
        
        if gpu_percent < 20:
            issues.append(f"🔴 Low GPU memory usage ({gpu_percent:.1f}%) - Consider increasing batch size")
        
        if gpu_percent > 95:
            issues.append(f"🟡 High GPU memory usage ({gpu_percent:.1f}%) - Risk of OOM")
        
        # Verificar utilização GPU
        gpu_util = stats.get('gpu_utilization', 0)
        if gpu_util < 50:
            issues.append(f"🔴 Low GPU utilization ({gpu_util:.1f}%) - Possible data loading bottleneck")
        
        # Verificar CPU
        cpu_percent = stats['cpu_percent']
        if cpu_percent > 90:
            issues.append(f"🟡 High CPU usage ({cpu_percent:.1f}%) - May affect data loading")
        
        # Diagnóstico
        if issues:
            print(f"\n⚠️ GPU Performance Issues Detected:")
            for issue in issues:
                print(f"   {issue}")
            
            print(f"\n💡 Optimization Suggestions:")
            if gpu_percent < 20:
                print(f"   - Increase BATCH_SIZE from {Config.BATCH_SIZE} to {Config.BATCH_SIZE * 2}")
                print(f"   - Enable MIXED_PRECISION training")
            
            if gpu_util < 50:
                print(f"   - Increase NUM_WORKERS from {Config.NUM_WORKERS} to {Config.NUM_WORKERS * 2}")
                print(f"   - Enable PIN_MEMORY and PERSISTENT_WORKERS")
                print(f"   - Increase PREFETCH_FACTOR")
        else:
            print("✅ GPU performance looks good!")
    
    def plot_performance_history(self, save_path=None):
        """Plotar histórico de performance"""
        if not self.gpu_stats:
            print("⚠️ No data to plot")
            return
        
        # Processar dados
        timestamps = [s['timestamp'] for s in self.gpu_stats]
        start_time = timestamps[0]
        times = [(t - start_time) / 60 for t in timestamps]  # Converter para minutos
        
        cpu_usage = [s['cpu_percent'] for s in self.gpu_stats]
        ram_usage = [s['ram_percent'] for s in self.gpu_stats]
        
        # Criar figura
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GPU Training Performance Monitor', fontsize=16)
        
        # CPU Usage
        axes[0, 0].plot(times, cpu_usage, 'b-', linewidth=2)
        axes[0, 0].set_title('CPU Usage (%)')
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('CPU %')
        axes[0, 0].grid(True, alpha=0.3)
        
        # RAM Usage
        axes[0, 1].plot(times, ram_usage, 'g-', linewidth=2)
        axes[0, 1].set_title('RAM Usage (%)')
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('RAM %')
        axes[0, 1].grid(True, alpha=0.3)
        
        if torch.cuda.is_available():
            gpu_memory = [s['gpu_memory_allocated'] for s in self.gpu_stats]
            gpu_util = [s.get('gpu_utilization', 0) for s in self.gpu_stats]
            
            # GPU Memory
            axes[1, 0].plot(times, gpu_memory, 'r-', linewidth=2)
            axes[1, 0].set_title('GPU Memory Usage (GB)')
            axes[1, 0].set_xlabel('Time (minutes)')
            axes[1, 0].set_ylabel('Memory (GB)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # GPU Utilization
            axes[1, 1].plot(times, gpu_util, 'orange', linewidth=2)
            axes[1, 1].set_title('GPU Utilization (%)')
            axes[1, 1].set_xlabel('Time (minutes)')
            axes[1, 1].set_ylabel('GPU %')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Performance plot saved to {save_path}")
        
        plt.show()

def profile_dataloader_performance(data_loader, device, num_batches=10):
    """Profilear performance do DataLoader"""
    print(f"\n🔍 Profiling DataLoader Performance ({num_batches} batches)")
    print("-" * 50)
    
    times = {
        'data_loading': [],
        'gpu_transfer': [],
        'total_batch': []
    }
    
    for i, (images, targets) in enumerate(data_loader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        
        # Simular transferência para GPU
        transfer_start = time.time()
        if device.type == 'cuda':
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            torch.cuda.synchronize()
        else:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        transfer_time = time.time() - transfer_start
        total_time = time.time() - batch_start
        data_loading_time = total_time - transfer_time
        
        times['data_loading'].append(data_loading_time)
        times['gpu_transfer'].append(transfer_time)
        times['total_batch'].append(total_time)
        
        print(f"Batch {i+1}: Total={total_time:.3f}s, "
              f"Loading={data_loading_time:.3f}s, Transfer={transfer_time:.3f}s")
    
    # Estatísticas
    print(f"\n📊 DataLoader Performance Summary:")
    for phase, time_list in times.items():
        avg_time = np.mean(time_list)
        max_time = np.max(time_list)
        min_time = np.min(time_list)
        print(f"   {phase}: Avg={avg_time:.3f}s, Min={min_time:.3f}s, Max={max_time:.3f}s")
    
    # Diagnóstico
    avg_loading = np.mean(times['data_loading'])
    avg_transfer = np.mean(times['gpu_transfer'])
    
    if avg_loading > avg_transfer * 2:
        print(f"\n⚠️ Data loading is the bottleneck!")
        print(f"   Consider increasing NUM_WORKERS or enabling PERSISTENT_WORKERS")
    elif avg_transfer > avg_loading * 2:
        print(f"\n⚠️ GPU transfer is the bottleneck!")
        print(f"   Consider enabling PIN_MEMORY or using non_blocking=True")
    else:
        print(f"\n✅ DataLoader performance looks balanced!")

if __name__ == "__main__":
    # Exemplo de uso
    monitor = GPUMonitor()
    monitor.start_monitoring(interval=0.5)
    
    try:
        # Simular treinamento
        print("🔄 Simulating training...")
        time.sleep(10)
        
        # Imprimir stats
        monitor.print_current_stats()
        monitor.diagnose_gpu_issues()
        
    finally:
        monitor.stop_monitoring()
        monitor.plot_performance_history()
