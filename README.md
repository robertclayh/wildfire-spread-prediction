# Wildfire Spread Prediction (mNDWS)

Machine learning models for **next-day wildfire spread prediction** using multimodal geospatial, meteorological, and vegetation data.

This repository implements a complete pipeline and several deep learning models for wildfire segmentation based on the **Modified Next-Day Wildfire Spread (mNDWS)** dataset.

---
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
---

## Repository Structure
mNDWS/
│
├── mNDWS_DataPipeline.py # Data ingestion, TFRecord → NPZ conversion, preprocessing, dataset loader
├── mNDWS_LogRegModel.ipynb # Baseline logistic regression model
├── mNDWS_UNetModel.ipynb # Physics-enhanced UNet model
├── mNDWS_ResNet18UNet.ipynb # UNet with ResNet-18 encoder
├── mNDWS_EMA_Polyak.ipynb # EMA + Polyak comparison notebook
│
├── requirements.txt # Python dependencies
├── logreg_config.yaml # Configuration for logistic regression
├── unet_config.yaml # Configuration for UNet models
│
├── scratch/ # Intermediate outputs, temporary data, experimental results
├── outputs/ # Finalized plots, metrics, and generated data files
└── README.md # This README file


---
---

## 0. Set Up — 
---
`Requirements.txt` 

Contains information on packages needed to run code and their version number. 

---
`logreg_config.yaml`

Defines parameters needed for the logistic regression model including the seed, batch size, channels/hyperparameters selected as well as model, optimization and training specs. 

---
`logreg_config.yaml`

Defines parameters needed for the logistic regression model including the seed, batch size, channels/hyperparameters selected as well as model, optimization and training specs.

---
`unet_config.yaml`
Defines parameters needed for the unet model including the seed, batch size, channels/hyperparameters selected as well as loss, optimization and training specs.

Sample code to access a .yaml file and check output: 

```python
import yaml

with open('file_name.yaml', 'r') as file:
    data = yaml.safe_load(file)

print(data)
```


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

## 2. Models — 

`mNDWS_models.py`

Shared model + training utilities for the wildfire experiments.

Exposes helpers to build the data pipeline, logistic regression baseline, and
PhysicsPrior UNet bundle so notebooks and scripts can stay in sync.

Example:
```
>>> import mNDWS_models as models
>>> train_ds, val_ds, test_ds, *_ = models.pipeline_hookup(BATCH_SIZE=8)
>>> lr_model, *_ = models.PixelLogReg_outputs(train_ds, meanC=models.meanC, stdC=models.stdC,
...                                          train_loader=models.train_loader, device=models.device) 
```

## 3. Logistic Regression 

`train_logreg.py`
 
Train the logistic-regression wildfire baseline.

Launch this script from the repo root to run the shared data pipeline,
fit the pixel-level logistic regression model for a configurable number of
epochs, and save the trained weights plus channel stats to an artifact file.
Use `--config path/to/config.yaml` to pull hyperparameters from YAML or rely
on the CLI flags (epochs, batch size, output path, checkpoint resume).

Example
-------
Train for five epochs with CLI defaults and save to a scratch artifact:

    python train_logreg.py --epochs 5 --output outputs/logreg_smoke_test.pt

---
## 4. UNet

### 4.1 Physics-Enhanced UNet — 

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


### 4.2 EMA + Polyak Averaging — `mNDWS_EMA_Polyak.ipynb`

Compares three weight types:

- **RAW model**
- **EMA (Exponential Moving Average)**
- **Polyak (running average)**

### 4.3 `train_unet.py`

Train the PhysicsPrior UNet wildfire model.

Run this script from the repo root to reuse the shared mNDWS data pipeline,
fit the PhysicsPrior UNet with EMA/Polyak tracking, and save checkpoints plus
channel stats. Provide `--config path/to/config.yaml` for full control or rely
on the CLI defaults (epochs, batch size, output path, checkpoint resume).

Example
-------
Train for a single quick epoch using the default config fallback:

    python train_unet.py --epochs 1 --output outputs/unet_smoke_test.pt
"""
---
## 5. Results - `eval_models.py`
Unified evaluation entry point for the logistic regression and PhysicsPrior UNet models.

Can output the following information: 
- AP, F1, IOU, and thresholds for test and training 
- pr curve 
- history curves
- confusion matrices
- logistic regression plots
- unet plots
- learnable parameters
- avg. epoch wall time 
- epoch time stdev 
- training throughput
- peak gpu memory
- inference latency

Example
-------
Run evaluation for a saved UNet checkpoint and the logistic regression baseline:

    python eval_models.py --config unet_config.yaml --ckpt outputs/unet_final.pt --model-type unet
    python eval_models.py --config logreg_config.yaml --ckpt outputs/logreg_final.pt --model-type lr



