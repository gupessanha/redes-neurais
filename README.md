# ✋ Rock Paper Scissors Detection using Faster R-CNN

Sistema de detecção de gestos manuais em tempo real utilizando Redes Neurais Convolucionais (CNN) para reconhecer os gestos **Pedra**, **Papel** e **Tesoura** através de webcam.

## 📖 Sobre o Projeto

Este projeto utiliza a arquitetura **Faster R-CNN com backbone ResNet-50 FPN** para realizar detecção de objetos em imagens e vídeo em tempo real.

O objetivo é identificar os gestos do jogo **Pedra, Papel e Tesoura**, utilizando técnicas de **Transfer Learning** e um modelo pré-treinado no dataset COCO.

Após o treinamento, o sistema é capaz de detectar e classificar os gestos manualmente realizados diante da câmera, retornando a classe identificada e sua respectiva caixa delimitadora (*bounding box*).

---

## 🚀 Tecnologias Utilizadas

* Python 3.10
* PyTorch
* Torchvision
* OpenCV
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook / Google Colab
* CUDA (opcional para treinamento em GPU)

---

## 🧠 Arquitetura Utilizada

O projeto foi desenvolvido utilizando:

* **Faster R-CNN**
* **ResNet-50**
* **Feature Pyramid Network (FPN)**

Fluxo da arquitetura:

1. Extração de características pela ResNet-50.
2. Geração de regiões candidatas através da Region Proposal Network (RPN).
3. Aplicação do RoI Align.
4. Classificação dos objetos detectados.
5. Ajuste das bounding boxes.
6. Retorno das detecções com suas respectivas probabilidades.

---

## 📂 Dataset

Foi utilizado o dataset:

**Rock Paper Scissors SXSW: Hand Gesture Detection**

Disponível em:

https://www.kaggle.com/datasets/lmonkeyrock/paper-rock-scissors

### Classes

O modelo foi treinado para identificar:

* Pedra
* Papel
* Tesoura
* Default (fundo/ausência de objeto relevante)

### Quantidade de Dados

Mais de **6.000 imagens anotadas** foram utilizadas durante o treinamento.

---

## 🔍 Tratamento e Validação dos Dados

Antes do treinamento foram realizadas etapas de:

* Análise de distribuição das classes
* Remoção de dados duplicados
* Identificação de imagens corrompidas
* Verificação de bounding boxes inválidas
* Correção de inconsistências nas anotações
* Validação visual das anotações

---

## ⚙️ Treinamento

Foi aplicada a técnica de **Transfer Learning**, utilizando pesos pré-treinados da Faster R-CNN.

### Hiperparâmetros

| Parâmetro     | Valor         |
| ------------- | ------------- |
| Otimizador    | SGD           |
| Learning Rate | 0.005         |
| Momentum      | 0.9           |
| Épocas        | 1 e 10        |
| Backbone      | ResNet-50 FPN |

---

## 📊 Resultados

### Comparação entre Treinamento de 1 e 10 Épocas

| Métrica       | 1 Época | 10 Épocas |
| ------------- | ------- | --------- |
| mAP@0.50      | 48.8%   | 92.1%     |
| mAP@0.50:0.95 | 25.5%   | 68.9%     |
| mAP@0.75      | 21.0%   | 78.0%     |
| Loss Final    | 0.785   | 0.241     |

### Principais Conclusões

* O treinamento com apenas 1 época mostrou-se insuficiente para convergência do modelo.
* Com 10 épocas foi observado um ganho significativo em todas as métricas.
* Houve redução de aproximadamente 70% na função de perda.
* O modelo apresentou excelente capacidade de generalização para o problema proposto.

---

## 🎥 Inferência em Tempo Real

Após o treinamento, o modelo pode ser utilizado em tempo real utilizando uma webcam.

O sistema:

* Captura os frames da câmera;
* Executa a inferência utilizando Faster R-CNN;
* Identifica os gestos detectados;
* Exibe as bounding boxes e probabilidades de classificação.

---

## 📁 Estrutura do Projeto

```bash
project/
│
├── dataset/
│   ├── images/
│   ├── annotations.csv
│
├── notebooks/
│   └── treinamento.ipynb
│
├── models/
│   └── faster_rcnn_rps.pkl
│
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│
├── results/
│   ├── metrics/
│   ├── figures/
│
└── README.md
```

---

## ▶️ Como Executar

### 1. Clone o Repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git

cd seu-repositorio
```

### 2. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 3. Execute o Treinamento

```bash
python train.py
```

### 4. Execute a Inferência em Tempo Real

```bash
python inference.py
```

---

## 👨‍💻 Autores

* Gustavo Oliveira Pessanha da Silva
* Mikaela Rikberg Alves

Universidade Federal do Rio de Janeiro (UFRJ)

---

## 📚 Referências

* Ren et al. (2015) – Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.
* He et al. (2016) – Deep Residual Learning for Image Recognition.
* Lin et al. (2014) – Microsoft COCO: Common Objects in Context.
* Lu et al. (2021) – Review on Convolutional Neural Network Applied to Plant Leaf Disease Classification.
