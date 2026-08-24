# TerraMind Adversarial Robustness

Official PyTorch implementation for the paper.

This repository contains the code to evaluate the feature-space vulnerability of the TerraMind Vision Transformer (a Foundation Model for Earth Observation) against $L_\infty$-bounded adversarial perturbations. 

## Setup and Data

Install the required dependencies (tested on Python 3.12 and PyTorch 2.11):
```bash
pip install -r requirements.txt
```

## Data Preparation

To facilitate out-of-the-box reproducibility, the 20 preprocessed multispectral patches (representing urban, water, and forest topologies) used in our experiments are directly included in the data/patches/ directory as ready-to-use PyTorch tensors and .png visualizations.

## Repository Layout
```
terramind-adversarial-robustness/
├── data/
│   ├── patches/      # Extracted multispectral patches (.pt / .png)
│   └── tensors/      # Generated adversarial noise tensors (.pt)
├── notebooks/        # Evaluation notebooks
├── plots/            # Exported figures
├── results/          # Output metrics (.csv)
└── src/              # Core modules (attacks, metrics, data loading)
```
## Reproducing the Results

The core optimization algorithms (PGD, FGSM) and spectral masking functions are located in src/attacks.py. The experiments are separated into standalone notebooks for clarity:

* 01_Baseline_Evaluation.ipynb: Compares FGSM and PGD optimization in the latent space.

* 02_Scaling_Laws.ipynb: Evaluates robustness across the TerraMind model scales (Tiny to Large).

* 03_Transferability.ipynb: Analyzes black-box adversarial transferability between different model capacities.

* 04_Spectral_Targeting.ipynb: Implements targeted attacks on specific physical indices (e.g., NDVI, NDWI).
