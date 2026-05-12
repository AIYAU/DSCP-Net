# DSD-MoE: Dual-Space Structured and Calibrated Mixture-of-Experts for Open-Set Hyperspectral Image Classification
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official implementation of **DSD-MoE**

## Overview

Given a hyperspectral image where only a subset of land-cover classes are known during training, DSD-MoE performs **closed-set classification** for known classes and **open-set detection** for unknown classes at test time.

## Quick Start

### Single-Dataset Training

```bash
# Default: mainline_simple variant (distance-only score + global calibration)
python train.py --dataset PaviaU --epochs 200

# With custom hyperparameters
python train.py --dataset Salinas --epochs 200 --batch_size 64 --lr 1e-4
```

### Supported Datasets

| Dataset | Classes | Known | Unknown | Spatial Size | Bands |
|---------|---------|-------|---------|-------------|-------|
| PaviaU  | 9       | 1–8   | 9       | 610×340     | 103   |
| Pavia   | 9       | 1–8   | 9       | 1096×715    | 102   |
| Salinas | 16      | 1–5,7–12 | 6,13–16 | 512×217 | 204   |
| IP (Indian Pines) | 16 | 1–8   | 9–16    | 145×145     | 200   |
| HT (Houston) | 15 | 1–12 | 13–15   | 349×1905    | 144   |
| LongKou | 9       | 1–8   | 9       | 550×400     | 270   |


