# DECTNet — Detail Enhanced CNN-Transformer Network for Single-Image Deraining

> **DECTNet** is a hybrid CNN–Transformer architecture designed for single-image deraining.
> It combines fine-grained local detail extraction (CNN-based) with global context modeling (Transformer-based) and introduces modules to preserve and recover high-frequency details.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Key Contributions](#key-contributions)
* [Model Architecture](#model-architecture)
* [Results Summary](#results-summary)
* [Repository Structure](#repository-structure)
* [Getting Started](#getting-started)

  * [Requirements](#requirements)
  * [Install](#install)
  * [Quick Start — Training](#quick-start--training)
  * [Quick Start — Inference / Test](#quick-start--inference--test)
* [Datasets & Evaluation](#datasets--evaluation)
* [Implementation Details & Hyperparameters](#implementation-details--hyperparameters)
* [Ablation Studies & Insights](#ablation-studies--insights)
* [Citing this Work](#citing-this-work)
* [Authors & Contact](#authors--contact)
* [License](#license)

---

# Project Overview

DECTNet is built to tackle the challenge of rain streak removal from single images while preserving fine textures and structural details. It addresses common trade-offs between CNNs (good at local detail) and Transformers (good at global context) by combining both in a staged design and adding detail-focused modules.

Core high-level flow:

1. Shallow feature extraction (conv)
2. Local information extraction — stacked **ERFDB** blocks
3. Global information extraction — stacked **DASTB** blocks
4. Multi-scale detail enhancement — **MS-ERFDB**
5. Fusion of multi-scale local + global features
6. Detail recovery — more **ERFDB** blocks
7. Reconstruction + global residual

---

# Key Contributions

* **ERFDB (Enhanced Residual Feature Distillation Block)**
  Progressive feature distillation with mixed spatial & channel attention and channel-enhanced layers for detailed local feature learning.

* **DASTB (Dual Attention Spatial Transformer Block)**
  Transformer block augmented with a spatial attention path and an IRB-inspired FFN for preserving spatial structure while modeling long-range dependencies.

* **MS-ERFDB (Multi-Scale ERFDB)**
  Multi-resolution processing (e.g., 1×, 0.5×, 0.25×) before fusion to handle rain streaks at different scales — yields a significant PSNR gain in experiments.

* **Negative SSIM Loss**
  Uses `L = - mean(SSIM)` as the objective to improve structural fidelity and convergence for restoration tasks.

---

# Model Architecture

```
Input Image (H×W×3)
    ↓
Shallow 3×3 Conv → Shallow Features
    ↓
Stage 1: N × ERFDB (local extraction)
    ↓
Stage 2: L × DASTB (global extraction)
    ↓
MS-ERFDB (multi-scale ERFDB)  ← (resized inputs × ERFDBs → upsample → merge)
    ↓
Fuse Block (adaptive GAP + GMP attention) → fused features
    ↓
Stage 3: N × ERFDB (detail recovery)
    ↓
Recon 3×3 Conv → Residual image + Input → Final derained image
```

* Typical configuration used in experiments: `N = 3` (ERFDBs per local/detail stage), `L = 5` (DASTBs).
* Patch size for Transformer (DASTB) experiments: 8.

---

# Results Summary

> Quantitative highlights reported in the analysis:

* **Rain200H (comparison with MS-ERFDB)**

  * Original DECTNet: **PSNR 39.06 dB / SSIM 0.9870**
  * DECTNet + MS-ERFDB: **PSNR 40.33 dB / SSIM 0.9870**

* **Rain100L**

  * Reported: **PSNR 38.94 dB / SSIM 0.9571**

* **Generalization**

  * Low-light enhancement and desnowing benchmarks showed competitive or state-of-the-art performance (e.g., Snow100K PSNR 32.28 / SSIM 0.95 in reported experiments).

---

# Repository Structure

```
├── decent.ipynb          # Jupyter notebook (implementation & experiments)
├── Report (2).pdf        # In-depth analysis & experimental results (uploaded)
├── README.md             # This file (raw markdown)
├── requirements.txt      # Python dependencies (PyTorch, etc.)
├── configs/              # Experiment config files (yaml/json)
├── datasets/             # dataset helper scripts / download helpers
├── models/
│   ├── dectnet.py        # DECTNet model (ERFDB, DASTB, MS-ERFDB)
│   ├── modules.py        # building blocks (attention, FFN, Fuse block)
│   └── loss.py           # negative SSIM and metric helpers
├── train.py              # training entrypoint
├── test.py               # inference and evaluation scripts
└── utils/
    ├── metrics.py
    └── transforms.py
```

---

# Getting Started

## Requirements

* Python 3.8+
* PyTorch 1.10+ (or latest stable)
* torchvision
* numpy, Pillow, scikit-image (for SSIM), tqdm

A sample `requirements.txt`:

```
torch>=1.10
torchvision
numpy
Pillow
scikit-image
tqdm
opencv-python
```

## Install

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start — Training

> Example command (adjust `--dataset`, `--epochs`, and paths as needed):

```bash
python train.py \
  --config configs/dectnet_rain100l.yaml \
  --dataset Rain100L \
  --epochs 150 \
  --batch-size 16 \
  --lr 5e-4 \
  --save-dir checkpoints/dectnet_rain100l
```

`train.py` should handle dataset loading (patch cropping to 128×128), optimizer (Adam with β1=0.9, β2=0.99), scheduler (optional), and model checkpointing.

## Quick Start — Inference / Test

```bash
python test.py \
  --model checkpoints/dectnet_best.pth \
  --input ./samples/rainy.jpg \
  --output ./results/derained.jpg
```

The test script should:

* Load the trained model
* Run forward pass on the input images (or directory)
* Save outputs and compute PSNR / SSIM if ground truth is provided

---

# Datasets & Evaluation

Recommended datasets used in experiments:

* **Rain100L**, **Rain200H**, **Rain1400** (synthetic deraining datasets)
* **Snow100K** (for desnowing generalization)
* **LOL (v1 & v2)** (for low-light enhancement generalization)

Evaluation metrics:

* **PSNR** (Peak Signal-to-Noise Ratio)
* **SSIM** (Structural Similarity Index)

Loss used for training in reported experiments:

* **Negative SSIM**: `L = -mean(SSIM(prediction, ground_truth))`

---

# Implementation Details & Hyperparameters

* **Optimizer**: Adam, `β1 = 0.9`, `β2 = 0.99`
* **Learning rate**: `5e-4` (with optional scheduler)
* **Input patch size**: `128 × 128` (random crops during training)
* **Epochs**: `150` (deraining baseline)
* **Batch size**: tune based on GPU memory (e.g., 8–32)
* **Model size**: \~1.51M parameters (reported for DECTNet baseline)
* **Transformer patch size** (for DASTB): `8` (patch-wise self-attention experiments)

---

# Ablation Studies & Insights

Reported ablations and findings:

1. **Stage Ordering**
   Local-first (ERFDB → DASTB) performs better than the reversed order.

2. **ERFDB Components**
   Removing mixed attention or channel-enhanced layers degrades performance substantially.

3. **DASTB Components**
   Spatial attention + IRB-style FFN help preserve structure and improve detail over a plain Transformer block.

4. **MS-ERFDB**
   Processing multiple scales before fusion yields +1.27 dB PSNR improvement in reported experiments, with SSIM preserved.

5. **Loss Function**
   Negative SSIM loss converged faster and yielded better SSIM compared to MSE in the experiments.

---

# Citing this Work

If you use this project or the ideas from it in your research, please cite:

```bibtex
@article{wang2025dectnet,
  title={DECTNet: A detail enhanced CNN-Transformer network for single-image deraining},
  author={Wang, L. and Gao, G.},
  journal={Cognitive Robotics},
  year={2025},
  volume={5},
  pages={48--60},
  doi={10.1016/j.cogr.2024.12.002}
}
```

And optionally the analysis report by the implementers:

```bibtex
Tiwari, A., & Bajaj, M. (2025). In-Depth Analysis of DECTNet: A Detail Enhanced CNN-Transformer Network for Single-Image Deraining. (Report included in repository)
```

---

# Authors & Contact

* **Abhay Tiwari** — [ifi2022024@iiita.ac.in](mailto:ifi2022024@iiita.ac.in) — IIIT Allahabad (Information Technology)

---

# License

This repository is released under the **MIT License** — see `LICENSE` for details.

---

## Notes

* The repository includes `decent.ipynb` (notebook) and `Report (2).pdf` (detailed analysis and experimental results). Use the notebook to inspect model implementation and reproduce experiments.
* Tweak hyperparameters (batch size, learning rate schedule, number of blocks) to match available hardware and improve results on custom datasets.

---
