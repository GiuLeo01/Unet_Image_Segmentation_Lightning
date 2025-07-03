# U-Net Image Segmentation with Lightning

A PyTorch Lightning implementation of U-Net for human image segmentation tasks.

## Overview

This project implements a U-Net convolutional neural network for semantic segmentation, and trains it on the **Human Segmentation MADS Dataset** for human body segmentation in images.

## Architecture

- **Encoder**: 4 encoder blocks with progressive feature extraction (64->128->256->512 filters)
- **Bottleneck**: 1024 filters for deep feature processing
- **Decoder**: 4 decoder blocks with skip connections for precise localization
- **Output**: Single channel binary segmentation mask

## Training

The model uses:
- Adam optimizer (lr=1e-3)
- Dice loss function
- Pixel-wise accuracy and F1-score metrics
- Automatic mixed precision support via Lightning

## Dataset

Trained on the **Human Segmentation MADS Dataset** - tapakah68. (2023). Human Segmentation MADS Dataset, 1192 images. Kaggle. [https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset)
