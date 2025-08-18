# SemiSupCon: A Unified Contrastive Loss for Self-Training

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of **A Unified Contrastive Loss for Self-Training**, a classification semi-supervised learning approach that combines the strengths of pseudo-labeling methods (like Fixmatch) and adapt it to contrastive learning framework by using prototype-based representations. This implementation is based on the excellent [USB (Unified SSL Benchmark)](https://github.com/microsoft/Semi-supervised-learning) framework from Microsoft Research, which provides a comprehensive benchmark for semi-supervised learning algorithms. We integrated our new `SemiSupCon` loss into the library, making it available within the USB framework and allowing straightforward, reproducible comparisons against existing semi-supervised learning baselines.


## 📚 Paper

**A Unified Contrastive Loss for Self-Training**

*This is the official implementation of our paper published at ECML 2024 :*

```bibtex
@inproceedings{Gauffre2024Unified,
  author = {Gauffre, A. and Horvat, J. and Amini, M. R.},
  title = {A Unified Contrastive Loss for Self-training},
  booktitle = {Machine Learning and Knowledge Discovery in Databases. Research Track and Demo Track. ECML PKDD 2024},
  series = {Lecture Notes in Computer Science},
  volume = {14948},
  publisher = {Springer, Cham},
  year = {2024},
  doi = {10.1007/978-3-031-70371-3_1}
}
```


## 🚀 Overview

SemiSupCon extends the popular FixMatch algorithm by introducing:
- **Contrastive Learning**: Leverages supervised contrastive loss to learn better feature representations
- **Prototype-based Pseudo-labeling**: Uses learned class prototypes for more robust pseudo-label generation
- **Adaptive Weighting**: Implements weighting schemes for different types of data (labeled, confident unlabeled, unconfident unlabeled)
- **Multi-view Consistency**: Maintains consistency across different augmented views of the same data



## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- CUDA (for GPU training)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/semisupcon.git
cd semisupcon

# Install dependencies
pip install -r requirements.txt

# For GPU support
pip install -r requirements_JZ_pytorch-gpu-1.13.0.txt
```

## 🚀 Quick Start

### Basic Training
```bash
python train.py --c ./config/classic_cv/semisupcon/semisupcon_cifar100_2500_0.yaml
```

### Configuration
The algorithm supports various configurations through YAML files:
- Dataset selection (CIFAR-10/100, STL-10, ImageNet)
- Network architectures (WideResNet, ResNet, Vision Transformers)
- Loss function variants
- Training hyperparameters

### Key Parameters
- `--algorithm`: Choose `semisupconproto` for the main algorithm
- `--loss`: Select loss function variant
- `--lambda_proto`: Weight for prototype loss
- `--lambda_yup`: Weight for confident unlabeled samples
- `--lambda_ydown`: Weight for unconfident unlabeled samples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.










