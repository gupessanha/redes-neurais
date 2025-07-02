"""
Script de teste rápido para verificar otimizações de GPU
Testa as otimizações implementadas sem treinar o modelo completo
"""
import torch
import time
from config import Config
from data_utils import DataLoader
from model_utils import ModelManager
from gpu_monitor import GPUMonitor, profile_dataloader_performance

def test_gpu_optimizations():
    """Testar otimizações de GPU implementadas"""
    print("="*60)
    print("TESTE DE OTIMIZAÇÕES GPU")
    print("="*60)
    
    # Configurar dispositivo
    device = Config.get_device()
    Config.setup_gpu_optimizations()
    
    print(f"\n📊 Configurações Otimizadas:")
    print(f"   - Batch Size: {Config.BATCH_SIZE}")
    print(f"   - Workers: {Config.NUM_WORKERS}")
    print(f"   - Pin Memory: {Config.PIN_MEMORY}")
    print(f"   - Prefetch Factor: {Config.PREFETCH_FACTOR}")
    print(f"   - Persistent Workers: {Config.PERSISTENT_WORKERS}")
    print(f"   - Mixed Precision: {Config.MIXED_PRECISION}")
    
    # Inicializar componentes
    print(f"\n🔄 Inicializando componentes...")
    data_loader = DataLoader()
    model_manager = ModelManager(device)
    
    # Carregar dados pequenos para teste
    print(f"\n📁 Carregando dados de teste...")
    df_image_info = data_loader.load_annotations(Config.TRAIN_DATA_PATH)
    
    # Usar apenas primeiras 50 imagens para teste rápido
    df_test = df_image_info.head(100)  # Pequeno subset para teste
    print(f"📊 Usando {len(df_test)} anotações para teste")
    
    # Carregar dados
    image_tensors, targets = data_loader.load_training_data(df_test, Config.TRAIN_IMAGES_PATH)
    print(f"✅ Carregadas {len(image_tensors)} imagens")
    
    if len(image_tensors) == 0:
        print("❌ Nenhuma imagem carregada. Verifique os caminhos dos arquivos.")
        return
    
    # Criar data loaders otimizados
    print(f"\n🚀 Criando DataLoaders otimizados...")
    train_loader, valid_loader = data_loader.create_data_loaders(image_tensors, targets)
    
    # Testar performance do DataLoader
    print(f"\n⚡ Testando Performance do DataLoader...")
    profile_dataloader_performance(train_loader, device, num_batches=5)
    
    # Criar modelo para teste
    print(f"\n🧠 Criando modelo para teste...")
    model = model_manager.create_model()
    model = model.to(device)
    
    # Iniciar monitoramento
    monitor = GPUMonitor()
    monitor.start_monitoring(interval=0.5)
    
    try:
        # Testar algumas iterações de treinamento
        print(f"\n🔄 Testando {3} épocas de treinamento...")
        
        # Configurar optimizer
        optimizer = model_manager.setup_optimizer(model)
        
        # Simular treinamento com mixed precision se disponível
        use_amp = Config.MIXED_PRECISION and torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
            print("⚡ Usando Mixed Precision (AMP)")
        
        total_batches = 0
        start_time = time.time()
        
        for epoch in range(3):  # Apenas 3 épocas para teste
            print(f"\n--- Época {epoch + 1}/3 ---")
            model.train()
            
            for batch_idx, (images, targets) in enumerate(train_loader):
                if batch_idx >= 3:  # Apenas 3 batches por época
                    break
                
                batch_start = time.time()
                
                # Transferir para GPU otimizada
                if device.type == 'cuda':
                    images = [img.to(device, non_blocking=True) for img in images]
                    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
                else:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass com AMP se disponível
                if use_amp:
                    with torch.cuda.amp.autocast():
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                    
                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    losses.backward()
                    optimizer.step()
                
                batch_time = time.time() - batch_start
                total_batches += 1
                
                # Mostrar stats atuais
                if device.type == 'cuda':
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f"  Batch {batch_idx + 1}: Loss={losses.item():.4f}, "
                          f"Time={batch_time:.3f}s, GPU={gpu_memory:.1f}GB")
                else:
                    print(f"  Batch {batch_idx + 1}: Loss={losses.item():.4f}, Time={batch_time:.3f}s")
        
        total_time = time.time() - start_time
        avg_batch_time = total_time / total_batches if total_batches > 0 else 0
        
        print(f"\n📊 Resultados do Teste:")
        print(f"   - Total de batches processados: {total_batches}")
        print(f"   - Tempo total: {total_time:.2f}s")
        print(f"   - Tempo médio por batch: {avg_batch_time:.3f}s")
        
        # Mostrar estatísticas finais
        monitor.print_current_stats()
        monitor.diagnose_gpu_issues()
        
        # Recomendações
        print(f"\n💡 Análise de Performance:")
        
        if device.type == 'cuda':
            current_stats = monitor.get_current_stats()
            gpu_used = current_stats['gpu_memory_allocated']
            gpu_total = current_stats['gpu_memory_total']
            gpu_percent = (gpu_used / gpu_total) * 100
            
            if gpu_percent < 30:
                suggested_batch = Config.BATCH_SIZE * 2
                print(f"   🔴 GPU subutilizada ({gpu_percent:.1f}%)")
                print(f"   💡 Sugestão: Aumentar BATCH_SIZE para {suggested_batch}")
            elif gpu_percent > 85:
                print(f"   🟡 GPU bem utilizada ({gpu_percent:.1f}%)")
                print(f"   ⚠️ Cuidado com out-of-memory se aumentar batch size")
            else:
                print(f"   ✅ GPU bem balanceada ({gpu_percent:.1f}%)")
            
            if avg_batch_time > 1.0:
                print(f"   ⚠️ Tempo por batch alto ({avg_batch_time:.3f}s)")
                print(f"   💡 Considere aumentar NUM_WORKERS ou habilitar otimizações")
            else:
                print(f"   ✅ Tempo por batch bom ({avg_batch_time:.3f}s)")
        
        # Comparação com configuração antiga
        print(f"\n📈 Comparação com Configuração Anterior:")
        print(f"   - Batch Size: 4 → {Config.BATCH_SIZE} ({Config.BATCH_SIZE/4:.1f}x maior)")
        print(f"   - Workers: 0 → {Config.NUM_WORKERS} (paralelização habilitada)")
        print(f"   - Pin Memory: False → {Config.PIN_MEMORY}")
        print(f"   - Mixed Precision: False → {Config.MIXED_PRECISION}")
        
        expected_speedup = (Config.BATCH_SIZE / 4) * 1.5  # Estimativa conservadora
        print(f"   🚀 Speedup esperado: ~{expected_speedup:.1f}x")
        
    except Exception as e:
        print(f"❌ Erro durante teste: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        monitor.stop_monitoring()
        
        # Limpeza
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"\n✅ Teste de otimizações concluído!")

if __name__ == "__main__":
    test_gpu_optimizations()
