# GA-PINN: Gradient-Aligned Physics-Informed Neural Network

A gradient-aligned physics-informed neural network for performance analysis of permanent magnet eddy current devices under complex operating conditions.

## 📖 Citation

If you use this code, please cite the following paper:

```bibtex
@article{WANG2026129915,
title = {Gradient-aligned physics-informed neural network for performance analysis of permanent magnet eddy current device under complex operating conditions},
journal = {Expert Systems with Applications},
volume = {299},
pages = {129915},
year = {2026},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2025.129915},
url = {https://www.sciencedirect.com/science/article/pii/S0957417425035304},
author = {Sihan Wang and Kai Wang and Peng Zeng and Yaguo Lei and Zhiping Wang and Bo Zhang},
keywords = {Surrogate modeling, Performance analysis, Physics-informed neural network, Permanent magnet eddy current device},
abstract = {The permanent magnet drive (PMD) is a non-contact mechanical power transmission device that demonstrates broad application prospects in industrial fields due to its exceptional torque control performance and structural reliability. However, with the increasing demand for real-time condition monitoring and the advancement of high-power-density PMD design, achieving efficient and accurate performance analysis has become a critical challenge. Artificial intelligence-driven digital twin approaches offer a novel perspective to address this issue, among which Physics-Informed Neural Networks (PINNs) exhibit significant potential by integrating data with physical knowledge. Nevertheless, under complex operating conditions, the skin effect of electromagnetic fields and the scale variation of solution domains in PMD severely limit the prediction accuracy and physical consistency of PINNs. To address the aforementioned challenges, this paper proposes a Gradient-Aligned Physics-Informed Neural Network (GA-PINN) method for performance analysis of PMD under complex conditions. GA-PINN incorporates electromagnetic governing equations and boundary conditions into the loss function to guide the model training process. By identifying and aligning the gradient information of the loss terms, it dynamically balances the model's constraint capacity across multi-scale solution domains. Additionally, a dynamic sampling strategy is introduced, which reconstructs the distribution probability of collocation points by identifying the gradient characteristics of the solution function, thereby addressing the mismatch in the global distribution of collocation points caused by localized high residuals. Through experimental validation across multiple operating conditions and structural parameters, GA-PINN achieves efficient and accurate performance analysis of PMD in complex scenarios, providing a reliable theoretical basis for its performance monitoring and optimization design. Dataset link: https://github.com/wongsihan/GA-PINN.}
}
```

## 🎯 Project Overview

This project implements 5 different PINN methods for electromagnetic field analysis and performance prediction of permanent magnet eddy current devices:

- **GA-PINN**: Gradient-Aligned Physics-Informed Neural Network
- **IDW-PINN**
- **LB-PINN**
- **PMC-PINN**
- **RAMAW-PINN**

## 🚀 Quick Start

### Requirements

- **Python**: 3.8+
- **PyTorch**: 1.9+
- **CUDA**: 11.0+ (recommended for GPU acceleration)
- **Other dependencies**: See `requirements.txt`

### Installation

```bash
# Clone the repository
git clone https://github.com/wongsihan/GA-PINN.git
cd GA-PINN

# Install dependencies
pip install -r requirements.txt

# Or use conda environment
conda create -n gapinn python=3.8
conda activate gapinn
pip install -r requirements.txt
```

## 🎮 Usage

### Run Individual Methods

```bash
# Run GA-PINN
python run_ga.py

# Run IDW-PINN
python run_idw.py

# Run LB-PINN
python run_lb.py

# Run PMC-PINN
python run_pmc.py

# Run RAMAW-PINN
python run_ramaw.py
```

### Run All Methods

```bash
# Run all PINN methods sequentially
python run_all.py
```

### Direct Source Execution

```bash
# Run main modules directly
python src/main.py          # GA-PINN
python src/main_idw.py      # IDW-PINN
python src/main_lb.py       # LB-PINN
python src/main_pmc.py      # PMC-PINN
python src/main_ramaw.py    # RAMAW-PINN
```

## 📊 Output Files

After running, the following files will be generated:

- **`results/`**: Contains all generated image files
  - `gamgp/`: GA-PINN result images
  - `idw/`: IDW-PINN result images  
  - `lb/`: LB-PINN result images
  - `pmc/`: PMC-PINN result images
  - `ramaw/`: RAMAW-PINN result images
- **`models/`**: Contains trained model files
- **Console output**: Training progress, loss values, and physical quantity calculations

## ⚠️ Notes

1. **GPU Memory**: Ensure sufficient GPU memory for training
2. **Training Time**: Training may take considerable time, please be patient
3. **Result Saving**: Generated images will be saved in the `results/` directory
4. **Model Files**: Trained models will be saved in the `models/` directory
5. **Environment**: Recommend using conda environment for dependency management