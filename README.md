# Understanding Data Efficiency on CIFAR-10

This repository contains code and results for a course project studying how training set size affects the performance of different machine learning models on the CIFAR-10 image classification task.

## Project Overview

We investigate the following question:

> How much labeled data is needed for different model families to reach useful accuracy on CIFAR-10?

To answer this, we train four models on stratified subsets of the CIFAR-10 training data at 5%, 10%, 25%, 50%, 75%, and 100% of the full training set (50,000 images), while keeping the 10,000-image test set fixed.

Models evaluated:

- Logistic Regression (linear baseline)
- Random Forest (100 trees)
- k-Nearest Neighbors (k = 5)
- 3-layer Convolutional Neural Network (CNN)

For each model and data fraction we record test accuracy and training time, and visualize the resulting learning curves.

## Files

- `colab_implementation.ipynb` — End-to-end notebook: data loading, preprocessing, model training, and plotting.
- `results.csv` — Final test accuracy and training time for each model and data fraction.
- `learning_curves.png` — Plot of test accuracy vs. training set size for all four models.
- `report.pdf` — Final project report (scientific-paper format, if included).

## How to Run

1. Open `colab_implementation.ipynb` in Google Colab (recommended) or locally with GPU support.
2. Run all cells in order.
3. The notebook will:
   - Download CIFAR-10
   - Create stratified subsets
   - Train all four models on each subset
   - Save `results.csv` and `learning_curves.png`

## Summary of Findings

- Random Forest is the most data-efficient model in low-data regimes (≤ 25% of training data).
- The CNN requires more data but achieves the highest accuracy once trained on ≥ 25–50% of the data.
- Logistic Regression and kNN plateau early and cannot fully exploit additional data on this task.


