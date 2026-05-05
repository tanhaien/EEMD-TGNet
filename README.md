# EEMD-TGNet: Hybrid Model for Solar Energy Forecasting

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4949.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/IEEE-Conference-red.svg" alt="IEEE">
</p>

A novel hybrid deep learning model combining **Ensemble Empirical Mode Decomposition (EEMD)** with **Temporal Convolutional Network - Gated Recurrent Unit (TCN-GRU)** architecture for accurate and efficient solar energy forecasting on edge devices.

## 📄 Paper

**Title**: A Novel Hybrid EEMD-TGNet Model for Accurate and Efficient Solar Energy Forecasting on Edge Devices

**Authors**: Toan Nguyen, Tuan-Anh Pham, Hao-Nguyen Nguyen, Van-Huy Tran, Nguyen Hoan Lam

**Affiliation**: Udata JSC, Hanoi, Vietnam

**Published**: IEEE Conference

📄 [Download Paper (PDF)](paper/TCN_GRU_IEEE_ACCESS_Final_Article.pdf)

## 🎯 Key Results

| Metric | Result |
|--------|--------|
| R² Score (GEFCom2014) | **91.1%** |
| MSE | **0.113** |
| Model Size Reduction | **75%** |
| Inference Speedup | **73%** |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EEMD-TGNet Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Input     │───►│    EEMD     │───►│   IMF 1     │        │
│  │  Solar Time │    │Decomposition│    │   (TCN-GRU) │        │
│  │    Series   │    └─────────────┘    └─────────────┘        │
│  │             │           │                  │                │
│  │             │           ▼                  ▼                │
│  │             │    ┌─────────────┐    ┌─────────────┐        │
│  │             │    │   IMF 2     │    │   IMF 3     │        │
│  │             │    │  (TCN-GRU)  │    │  (TCN-GRU)  │        │
│  │             │    └─────────────┘    └─────────────┘        │
│  │             │           │                  │                │
│  │             │           └──────────┬───────┘                │
│  │             │                      ▼                        │
│  │             │              ┌─────────────┐                 │
│  │             │              │Aggregation │                 │
│  │             │              │   (Sum)    │                 │
│  │             │              └─────────────┘                 │
│  │             │                      │                        │
│  └─────────────┘                      ▼                        │
│                           ┌─────────────┐                      │
│                           │   Output    │                      │
│                           │  Forecast   │                      │
│                           └─────────────┘                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │            Edge Optimization Pipeline                   │  │
│  │   ┌────────────┐    ┌────────────┐    ┌────────────┐    │  │
│  │   │  Original  │───►│  Pruning   │───►│Quantization│    │  │
│  │   │   Model    │    │  (40%)     │    │   (FP32    │    │  │
│  │   │ (80K params)│    │ (48K params)│    │   to INT8) │    │  │
│  │   └────────────┘    └────────────┘    └────────────┘    │  │
│  │        0.32MB            0.16MB            0.08MB        │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔬 Methodology

### 1. Ensemble Empirical Mode Decomposition (EEMD)
- Decomposes non-stationary solar time series into **Intrinsic Mode Functions (IMFs)**
- Uses noise-assisted decomposition with 100 ensemble realizations
- Autocorrelation-based IMF selection to remove noise components

### 2. TCN-GRU Hybrid Architecture (TGNet)
- **TCN**: Dilated causal convolutions for long-range temporal dependencies
- **GRU**: Efficient sequential modeling with gating mechanisms
- Each IMF gets a dedicated TCN-GRU model

### 3. Edge Optimization
- **Pruning**: Magnitude-based unstructured pruning (40% sparsity)
- **Quantization**: FP32 to INT8 post-training quantization
- Result: 75% memory reduction with <3% accuracy degradation

## 📁 Repository Structure

```
EEMD-TGNet/
├── paper/
│   ├── EEMD_TGNet_Solar_Forecasting_IEEE.tex    # LaTeX source
│   └── TCN_GRU_IEEE_ACCESS_Final_Article.pdf    # Published paper
│
├── src/
│   ├── 01_Data_Generation_EEMD_Decomposition.py  # Data preprocessing
│   ├── 02_TCN_GRU_Architecture_Training.py        # Model training
│   └── 03_Evaluation_Results_Analysis.py         # Evaluation
│
├── notebooks/
│   ├── 01_Data_Generation_EEMD_Decomposition.ipynb
│   ├── 02_TCN_GRU_Architecture_Training.ipynb
│   └── 03_Evaluation_Results_Analysis.ipynb
│
├── data/
│   ├── gefcom_prepared_data.pkl                  # GEFCom2014 dataset
│   └── alibaba_prepared_data.pkl                 # Alibaba competition
│
├── results/
│   ├── *.png                                     # Visualization figures
│   ├── *.csv                                     # Performance tables
│   └── training_results.pkl                      # Trained models
│
├── scripts/                                      # Utility scripts
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/tanhaien/EEMD-TGNet.git
cd EEMD-TGNet

# Create conda environment
conda env create -f environment.yml
conda activate eemd-tgnet

# Or using pip
pip install -r requirements.txt
```

### Running Experiments

```bash
# 1. Data Generation & EEMD Decomposition
jupyter notebook notebooks/01_Data_Generation_EEMD_Decomposition.ipynb

# 2. Model Training & Optimization
jupyter notebook notebooks/02_TCN_GRU_Architecture_Training.ipynb

# 3. Evaluation & Results Analysis
jupyter notebook notebooks/03_Evaluation_Results_Analysis.ipynb
```

## 📊 Performance Comparison

| Model | MSE | MAE | R² Score |
|-------|-----|-----|----------|
| LSTM | 0.0072 | 0.063 | 87.5% |
| GRU | 0.0061 | 0.052 | 89.8% |
| CNN-LSTM | 0.0058 | 0.049 | 90.2% |
| BiLSTM | 0.0055 | 0.046 | 90.5% |
| EEMD-BiLSTM | 0.0045 | 0.039 | 91.5% |
| **EEMD-TGNet (Proposed)** | **0.0039** | **0.034** | **91.1%** |

### Edge Optimization Results

| Model Version | Parameters | Memory | MSE |
|---------------|------------|--------|-----|
| Original | ~80,000 | 0.32 MB | 0.0039 |
| Pruned (40%) | ~48,000 | 0.16 MB | 0.0041 |
| Quantized (INT8) | ~48,000 | **0.08 MB** | 0.0041 |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | PyTorch |
| Signal Processing | EEMD (custom implementation) |
| Deep Learning | TCN, GRU |
| Visualization | Matplotlib, Seaborn |
| Development | Jupyter Notebook |
| Environment | Conda, Python 3.8+ |

## 📚 Datasets

1. **GEFCom2014** - Global Energy Forecasting Competition 2014
2. **Alibaba** - Solar power data from Alibaba cluster
3. **Chinese Grid** - Regional solar generation data

## 📝 Citation

```bibtex
@conference{eemd_tgnet,
  title={A Novel Hybrid EEMD-TGNet Model for Accurate and Efficient Solar Energy Forecasting on Edge Devices},
  author={Nguyen, T. and Pham, T.A. and Nguyen, H.N. and Tran, V.H. and Lam, N.H.},
  booktitle={IEEE Conference Proceedings},
  year={2025}
}
```

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🤝 Acknowledgments

- Udata JSC Research Department
- IEEE Conference Reviewers
- GEFCom2014 Competition Organizers