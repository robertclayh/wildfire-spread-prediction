# Wildfire Spread Prediction (mNDWS)

Machine learning models for **next-day wildfire spread prediction** using multimodal geospatial, meteorological, and vegetation data.

This repository implements a complete pipeline and several deep learning models for wildfire segmentation based on the **Modified Next-Day Wildfire Spread (mNDWS)** dataset.

---

## Project Overview

This project predicts **tomorrow's wildfire spread mask (64×64)** using:

- Meteorology (temperature, RH, wind components)
- Vegetation (NDVI)
- Fuels & drought (ERC, PDSI, BI)
- Terrain (slope, aspect, elevation)
- Landcover + population barriers
- Previous-day fire mask

It includes:

- A production-grade **data pipeline**
- Logistic Regression baseline
- **Physics-enhanced UNet**
- **ResNet-18 UNet**
- Exponential Moving Average (EMA) & Polyak averaging
- Full evaluation, threshold search, and visualization tools

---

## Repository Structure
mNDWS/
│
├── mNDWS_DataPipeline.py # Data ingestion, TFRecord→NPZ, preprocessing, dataset loader
├── mNDWS_LogRegModel.ipynb # Baseline logistic regression model
├── mNDWS_UNetModel.ipynb # Physics-enhanced UNet model
├── mNDWS_ResNet18UNet.ipynb # UNet with ResNet-18 encoder
├── mNDWS_EMA_Polyak.ipynb # EMA + Polyak comparison notebook
| 
├── requirements.txt
├── logreg_config.yaml 
├── unet_config.yaml 
|
├── scratch/ # contains information including intermediate results for transparency 
├── outputs/ # contains finalized plots and data files
└── README.md


---

## 1. Data Pipeline — `mNDWS_DataPipeline.py`

This script is the **foundation** of the project.

### What it does

- Detects/loads NPZ dataset  
- Converts TFRecords → NPZ if needed  
- Extracts & normalizes:

  - Fire masks
  - Temperature, RH, wind, gust
  - NDVI and vegetation indices
  - Fuel moisture indices
  - Slope & aspect
  - Barriers (population, water)
  - Landcover layers

- Builds the canonical **9-channel wildfire tile**
[prev_fire, u, v, temp, rh, ndvi, slope, aspect, barrier]


- Creates:
  - `WildfireDataset`
  - Train/Val/Test splits
  - WeightedRandomSampler
  - PyTorch DataLoaders  
  - Normalization statistics

---

## 2. Logistic Regression Baseline — `mNDWS_LogRegModel.ipynb`

A simple interpretability-first model:

- Loads NPZ tiles  
- Flattens features  
- Trains logistic regression  
- Reports:

  - Average Precision (AP)  
  - F1  
  - Precision–Recall curves  

This establishes a baseline before deep learning.

---

## 3. Physics-Enhanced UNet — `mNDWS_UNetModel.ipynb`

The main wildfire segmentation model.

### Key features

- **16-channel input** (9 raw + 3 geometry + 4 physics)
- **PhysicsPrior** module implementing:

  - Wind-aligned anisotropic spread kernel
  - Slope & aspect-driven spread
  - NDVI + RH damping
  - Barrier suppression

- UNet with:
  - SiLU activations
  - BatchNorm
  - ConvTranspose upsampling

### Training features

- Geometry-aware augmentation (corrects wind/aspect)
- Mixed precision (AMP)
- Cosine LR schedule
- Focal + Focal Tversky loss (50/50 mix)
- Max-F1 threshold selection
- Test-Time Augmentation (TTA)
- F1-by-tile-size diagnostics

---

## 4. ResNet-18 UNet — `mNDWS_ResNet18UNet.ipynb`

A stronger UNet variant using a ResNet-18 encoder.

Benefits:

- Better feature extraction
- Stronger context modeling
- Typically +2–4% AP over vanilla UNet

---

## 5. EMA + Polyak Averaging — `mNDWS_EMA_Polyak.ipynb`

Compares three weight types:

- **RAW model**
- **EMA (Exponential Moving Average)**
- **Polyak (running average)**

Notebook outputs:

```json
{
  "variant": "RAW | EMA | Polyak",
  "thr_plain": "...",
  "thr_tta": "...",
  "test_ap": "...",
  "test_f1": "..."
}
```

## 6. Visualization Tools

Across notebooks you get:

- False-color RGB composites  
- Prediction overlays  
- Ground-truth overlays  
- TP / FP / FN diff-maps  
- F1-by-fire-size plots  
- Interactive inspection utilities





