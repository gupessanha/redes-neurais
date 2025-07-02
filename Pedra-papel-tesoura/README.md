# Rock Paper Scissors Detection System

Sistema de detecção de gestos Pedra, Papel e Tesoura usando PyTorch e Faster R-CNN.

## 🔄 **Formato de Modelo Atualizado**

### ✅ **Novo: Formato PyTorch Padrão (.pth)**
- **Formato usado**: `torch.save(model.state_dict(), path)`
- **Padrão da indústria** para modelos PyTorch
- **Device-independent**: Funciona em qualquer máquina (CPU/CUDA)
- **Compatibilidade total** entre diferentes versões do PyTorch
- **Tamanho otimizado**: Salva apenas os pesos, não a arquitetura

### 🔧 **Benefícios do Novo Formato**
```python
# Salvamento (apenas state_dict)
torch.save(model.state_dict(), 'model.pth')

# Carregamento com map_location para compatibilidade
state_dict = torch.load('model.pth', map_location=device)
model.load_state_dict(state_dict)
```

**Vantagens:**
- ✅ **Padrão da indústria**: Formato oficial do PyTorch
- ✅ **Portabilidade**: Funciona em qualquer sistema operacional
- ✅ **Flexibilidade**: Permite diferentes devices no save/load
- ✅ **Segurança**: Não executa código Python arbitrário
- ✅ **Performance**: Carregamento mais rápido
- ✅ **Compatibilidade**: Entre versões do PyTorch

## Como executar com pipenv

### 1. Ativar o ambiente pipenv e executar o script:
```powershell
# No terminal PowerShell
pipenv shell
python dev_script.py
```

### 2. Executar diretamente com pipenv run:
```powershell
# Executa o script sem ativar o shell
pipenv run python dev_script.py
```

### 3. Testar novo formato PyTorch:
```powershell
# Testa compatibilidade do formato .pth
pipenv run python test_pytorch_format.py
```

### 4. Executar outros scripts:
```powershell
# Para usar a webcam diretamente
pipenv run python use_camera.py

# Para executar o notebook de desenvolvimento
pipenv run jupyter notebook dev_nb.ipynb
```

## Estrutura do Projeto

- `config.py` - Configurações do projeto (detecção CPU/CUDA, caminhos, parâmetros)
- `data_utils.py` - Utilitários para carregamento e processamento de dados
- `model_utils.py` - Gerenciamento de modelos (criação, carregamento, salvamento)
- `training_utils.py` - Utilitários de treinamento e avaliação
- `inference_utils.py` - Inferência em imagens e detecção via webcam
- `dev_script.py` - Script principal modularizado

## Funcionalidades

1. **Detecção automática de CPU/CUDA**
2. **Escolha de modelo**: Carregar modelo existente ou treinar novo
3. **Validação com imagens da pasta validation**
4. **Detecção em tempo real via webcam**
5. **Código modularizado para reutilização**

## Requisitos

- Python 3.8+
- PyTorch
- OpenCV
- Matplotlib
- Pandas
- Pillow
- scikit-learn

Instale as dependências com:
```powershell
pipenv install
```
