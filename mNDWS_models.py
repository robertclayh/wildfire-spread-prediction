# containing logistic and U-Net architectures
# =========================================================
# 0) Setup (Colab installs) + Utilities, Environment setup, deterministic seeds, device selection
# =========================================================

import os, math, random, glob, time, json, gc, pathlib, shutil 
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision.models import ResNet18_Weights
from sklearn.metrics import average_precision_score, precision_recall_curve

# Silence pipeline init logs unless overridden upstream
os.environ.setdefault("MNDWS_PIPELINE_SILENT", "1")
import mNDWS_DataPipeline as mndws_dp
WildfireDataset = mndws_dp.WildfireDataset
WildfirePaths = mndws_dp.WildfirePaths
CH_ORDER_BASE = mndws_dp.CH_ORDER_BASE
CH_ORDER_EXTRA = mndws_dp.CH_ORDER_EXTRA

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
    # Enable cuDNN autotuner only when CUDA is active
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

use_cuda = device.type == "cuda"
use_mps = device.type == "mps"
print("Device:", device)

workspace_root = pathlib.Path.cwd()
ART_ROOT = pathlib.Path(os.environ.get("ARTIFACTS_DIR", os.path.expanduser("~/wildfire_artifacts"))) / "resnet18_unet"
ART_ROOT.mkdir(parents=True, exist_ok=True)
print(f"Artifacts -> {ART_ROOT}")

# =============================================================
## 1) Data pipeline hookup + loaders + channel stats
## =============================================================

# --- Use shared mNDWS DataPipeline outputs for NPZ tiles and loaders ---
try:
    mndws_dp
except NameError:
    import os
    os.environ.setdefault("MNDWS_PIPELINE_SILENT", "1")
    import mNDWS_DataPipeline as mndws_dp
    WildfireDataset = mndws_dp.WildfireDataset
    WildfirePaths = mndws_dp.WildfirePaths
    CH_ORDER_BASE = mndws_dp.CH_ORDER_BASE
    CH_ORDER_EXTRA = mndws_dp.CH_ORDER_EXTRA

NPZ_ROOT = mndws_dp.NPZ_ROOT
print(f'Reusing NPZ tiles from pipeline at: {NPZ_ROOT}')

CHANNELS_FOR_MODEL = list(mndws_dp.USE_CHANNELS)  # adjust here if you want fewer features
paths = mndws_dp.WildfirePaths(NPZ_ROOT)

train_ds = mndws_dp.WildfireDataset(paths, split='train', max_samples=1200, channels=CHANNELS_FOR_MODEL)
val_ds   = mndws_dp.WildfireDataset(paths, split='eval',  max_samples=300,  channels=CHANNELS_FOR_MODEL)
test_ds  = mndws_dp.WildfireDataset(paths, split='test',  max_samples=300,  channels=CHANNELS_FOR_MODEL)

BATCH_SIZE = 8
MAX_PARALLEL_WORKERS = min(8, max(2, (os.cpu_count() or 4) // 2))
# Training benefits from multiple workers, but evaluation/test often run on memory-constrained nodes.
TRAIN_WORKERS = MAX_PARALLEL_WORKERS
EVAL_WORKERS = 0 if os.environ.get("MNDWS_EVAL_ALLOW_WORKERS", "0") == "0" else min(4, MAX_PARALLEL_WORKERS)
PIN_MEMORY = device.type == "cuda"

def make_balanced_loader(ds, *, batch_size, shuffle, upweight_positive, num_workers, pin_memory):
    sampler = None
    if upweight_positive:
        weights = []
        for f in ds.files:
            try:
                with np.load(f, mmap_mode="r") as z:
                    weights.append(5.0 if float(z["next_fire"].sum()) > 0 else 1.0)
            except Exception:
                weights.append(1.0)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    loader_kwargs = dict(batch_size=batch_size, pin_memory=pin_memory)
    if num_workers > 0:
        loader_kwargs["num_workers"] = num_workers
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    else:
        loader_kwargs["num_workers"] = 0
    if sampler is not None:
        return DataLoader(ds, sampler=sampler, shuffle=False, **loader_kwargs)
    return DataLoader(ds, shuffle=shuffle, **loader_kwargs)

train_loader = make_balanced_loader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, upweight_positive=True,
    num_workers=TRAIN_WORKERS, pin_memory=PIN_MEMORY
)
val_loader   = make_balanced_loader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, upweight_positive=False,
    num_workers=EVAL_WORKERS, pin_memory=PIN_MEMORY
)
test_loader  = make_balanced_loader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, upweight_positive=False,
    num_workers=EVAL_WORKERS, pin_memory=PIN_MEMORY
)
print(f"DataLoader workers -> train: {TRAIN_WORKERS}, val: {EVAL_WORKERS}, test: {EVAL_WORKERS}; pin_memory={PIN_MEMORY}")

meanC, stdC = mndws_dp.compute_channel_stats(train_ds, n_max_samples=4000, batch_size=32)
meanC, stdC = meanC.to(device), stdC.to(device)

print(f'Channels configured ({len(CHANNELS_FOR_MODEL)}): {CHANNELS_FOR_MODEL}')
print("Channel stats computed ->", meanC.shape, stdC.shape)

# =========================================================
# 2) Pixel Logistic Regression (1x1 conv) — mNDWS channel-aware, change number of epochs here
# =========================================================

class PixelLogReg(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.lin = nn.Conv2d(in_ch, 1, kernel_size=1, bias=True)
    def forward(self, x):
        return self.lin(x)  # logits (B,1,H,W)

def build_lr_input(X_raw0, mean=meanC, std=stdC):
    # Normalize using stats for the configured channels
    return (X_raw0 - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

@torch.no_grad()
def pos_weight_from_loader(loader, max_batches=100):
    total_pos = 0
    total = 0
    for i, b in enumerate(loader):
        y = b["y"]
        total_pos += y.sum().item()
        total     += y.numel()
        if i+1 >= max_batches: break
    pos = max(total_pos, 1.0)
    neg = max(total - total_pos, 1.0)
    return torch.tensor(neg / pos, dtype=torch.float32, device=device)

# Match model input to selected channels
def PixelLogReg_outputs(train_ds, meanC, stdC, train_loader, device):
    n_ch = len(train_ds.channels)
    assert meanC.numel() == n_ch and stdC.numel() == n_ch, "Stats must match channel count"

    lr_model = PixelLogReg(in_ch=n_ch).to(device)
    pw = pos_weight_from_loader(train_loader)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.AdamW(lr_model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("in_ch =", n_ch, "pos_weight =", float(pw))

    return lr_model, pw, criterion, optimizer
# change number of epochs here
#EPOCHS_LR = 50

# =============================================================
# 3) Model definition – ResNet-18 encoder + lightweight decoder
# =============================================================
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout=dropout)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNet18UNet(nn.Module):
    def __init__(self, in_ch=15, base_ch=64, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)
        if in_ch != 3:
            new_conv = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if weights is not None:
                with torch.no_grad():
                    new_conv.weight[:, :3] = resnet.conv1.weight
                    if in_ch > 3:
                        for c in range(3, in_ch):
                            new_conv.weight[:, c:c+1] = resnet.conv1.weight[:, (c % 3):(c % 3)+1]
            resnet.conv1 = new_conv
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.center = ConvBlock(512, 512, dropout=0.1)
        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 96)
        self.up0 = UpBlock(96, 64, base_ch)
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, 1, kernel_size=1)
        )
    def forward(self, x):
        s0 = self.stem(x)                # 64
        s1 = self.maxpool(s0)
        e1 = self.encoder1(s1)          # 64
        e2 = self.encoder2(e1)          # 128
        e3 = self.encoder3(e2)          # 256
        e4 = self.encoder4(e3)          # 512
        bottleneck = self.center(e4)
        d3 = self.up3(bottleneck, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        d0 = self.up0(d1, s0)
        out = F.interpolate(d0, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(out)
    
def ResNet18UNet_model(CHANNELS_FOR_MODEL):
    model = ResNet18UNet(in_ch=len(CHANNELS_FOR_MODEL), base_ch=96, pretrained=True).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} M")
    return model
