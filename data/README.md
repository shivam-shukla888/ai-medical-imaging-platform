# Dataset Directory

This directory is designated for the MedMNIST PneumoniaMNIST dataset file:
`pneumoniamnist.npz`

## How to obtain:
The dataset is part of the official MedMNIST v2 benchmark:
- Paper: "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification" (Yang et al., Nature Scientific Data 2023)
- Source: [MedMNIST](https://medmnist.com/)
- Filename: `pneumoniamnist.npz`
- Expected MD5: `28209eda62fecd6e6a2d98b1501bb15f`

Place `pneumoniamnist.npz` in this folder (`data/pneumoniamnist.npz`) or at the repository root. The configuration system automatically resolves both paths.

### Dataset Splits
- **Train split**: 4,708 images (1,214 Normal, 3,494 Pneumonia)
- **Validation split**: 524 images (135 Normal, 389 Pneumonia)
- **Test split**: 624 images (234 Normal, 390 Pneumonia)
- **Resolution**: 28 x 28 pixels, single grayscale channel, pixel range [0, 255]
