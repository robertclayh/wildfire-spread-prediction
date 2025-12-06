"""Shared model + training utilities for the wildfire experiments.

Exposes helpers to build the data pipeline, logistic regression baseline, and
PhysicsPrior UNet bundle so notebooks and scripts can stay in sync.

Example
-------
>>> import mNDWS_models as models
>>> train_ds, val_ds, test_ds, *_ = models.pipeline_hookup(BATCH_SIZE=8)
>>> lr_model, *_ = models.PixelLogReg_outputs(train_ds, meanC=models.meanC, stdC=models.stdC,
...                                          train_loader=models.train_loader, device=models.device)
"""

# =========================================================
# 0) Setup (Colab installs) + Utilities, Environment setup, deterministic seeds, device selection
# =========================================================

import os, math, random, glob, time, json, gc, pathlib, shutil 
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score, precision_recall_curve

# Silence pipeline init logs unless overridden upstream
os.environ.setdefault("MNDWS_PIPELINE_SILENT", "1")
import mNDWS_DataPipeline as mndws_dp
WildfireDataset = mndws_dp.WildfireDataset
WildfirePaths = mndws_dp.WildfirePaths
CH_ORDER_BASE = mndws_dp.CH_ORDER_BASE
CH_ORDER_EXTRA = mndws_dp.CH_ORDER_EXTRA

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(1337)

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

def configure_channels(
    base: Optional[List[str]] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[Set[str]] = None,
) -> List[str]:
    """Return an ordered channel list, optionally adding/removing entries for ablations."""
    base_list = list(base if base is not None else mndws_dp.USE_CHANNELS)
    exclude_set = set(exclude or [])
    ordered: List[str] = []
    seen: Set[str] = set()
    for name in base_list:
        if name in exclude_set:
            continue
        if name in seen:
            continue
        ordered.append(name)
        seen.add(name)
    if include:
        for name in include:
            if name not in seen:
                ordered.append(name)
                seen.add(name)
    return ordered


def pipeline_hookup(
    CHANNELS_FOR_MODEL: Optional[List[str]] = None,
    BATCH_SIZE: int = 16,
    include_channels: Optional[List[str]] = None,
    exclude_channels: Optional[Set[str]] = None,
):
    """Create datasets/loaders using full pipeline channels with easy ablation hooks."""
    channels = configure_channels(CHANNELS_FOR_MODEL, include_channels, exclude_channels)
    paths = mndws_dp.WildfirePaths(NPZ_ROOT)

    train_ds = mndws_dp.WildfireDataset(paths, split='train', max_samples=1200, channels=channels)
    val_ds   = mndws_dp.WildfireDataset(paths, split='eval',  max_samples=300,  channels=channels)
    test_ds  = mndws_dp.WildfireDataset(paths, split='test',  max_samples=300,  channels=channels)

    train_loader = mndws_dp.make_loader(train_ds, batch_size=BATCH_SIZE, upweight_positive=True)
    val_loader   = mndws_dp.make_loader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = mndws_dp.make_loader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    meanC, stdC = mndws_dp.compute_channel_stats(train_ds, n_max_samples=2000, batch_size=32)
    meanC, stdC = meanC.to(device), stdC.to(device)

    print(f'Channels configured ({len(channels)}): {channels}')
    print("Channel stats computed ->", meanC.shape, stdC.shape)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, meanC, stdC


# =========================================================
# 2) Pixel Logistic Regression (1x1 conv) — mNDWS channel-aware, change number of epochs here
# =========================================================

class PixelLogReg(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.lin = nn.Conv2d(in_ch, 1, kernel_size=1, bias=True)
    def forward(self, x):
        return self.lin(x)  # logits (B,1,H,W)

def build_lr_input(X_raw0, meanC, stdC):
    # Normalize using stats for the configured channels
    return (X_raw0 - meanC.view(1, -1, 1, 1)) / stdC.view(1, -1, 1, 1)

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
# 2b) Optional loss functions (Focal + Tversky) for segmentation heads
# =============================================================

class BinaryFocalLoss(nn.Module):
    """Focal loss wrapper around BCE-with-logits for class-imbalanced masks."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean", pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of none|mean|sum")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.type_as(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets,
                                                 pos_weight=self.pos_weight,
                                                 reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = alpha_t * (1.0 - pt).pow(self.gamma) * bce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class TverskyLoss(nn.Module):
    """Tversky loss (1 - index) tailored for binary masks using logits input."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.type_as(logits)
        probs = torch.sigmoid(logits)
        dims = tuple(range(1, probs.ndim))
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probs) * targets).sum(dim=dims)
        score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - score.mean()


class HybridFocalTverskyLoss(nn.Module):
    """Weighted sum of focal and Tversky losses (single forward for each)."""

    def __init__(
        self,
        focal_loss: BinaryFocalLoss,
        tversky_loss: TverskyLoss,
        focal_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.focal = focal_loss
        self.tversky = tversky_loss
        self.focal_weight = float(focal_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        fw = float(min(max(self.focal_weight, 0.0), 1.0))
        focal_term = self.focal(logits, targets)
        tversky_term = self.tversky(logits, targets)
        return fw * focal_term + (1.0 - fw) * tversky_term


def build_physics_loss(loss_type: str = "bce", *, pos_weight: Optional[torch.Tensor] = None,
                       focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                       tversky_alpha: float = 0.5, tversky_beta: float = 0.5,
                       focal_weight: float = 0.5,
                       reduction: str = "mean") -> nn.Module:
    """Factory for physics UNet losses (BCE, Focal, Tversky)."""

    loss_type = loss_type.lower()
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
    if loss_type == "focal":
        return BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                               reduction=reduction, pos_weight=pos_weight)
    if loss_type == "tversky":
        return TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
    if loss_type in {"focal_tversky", "hybrid"}:
        focal_loss = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                                     reduction=reduction, pos_weight=pos_weight)
        tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        return HybridFocalTverskyLoss(focal_loss, tversky_loss, focal_weight=focal_weight)
    raise ValueError(f"Unsupported loss_type '{loss_type}' (expected bce|focal|tversky)")

# =============================================================
# 3) Physics-prior UNet + EMA / Polyak trackers
# =============================================================

PHYSICS_BASE_CHANNELS = [
    "prev_fire", "u", "v", "temp", "rh", "ndvi", "slope", "aspect", "barrier"
]
OPTIONAL_PHYSICS_BASE_CHANNELS: Set[str] = set(PHYSICS_BASE_CHANNELS)


class PhysicsPrior(nn.Module):
    """Builds physics-inspired spread features used by the compact UNet."""

    def __init__(self, kernel_radius: int = 4, a0: float = 0.0, a1: float = 0.03,
                 a2: float = 0.02, a3: float = 0.7) -> None:
        super().__init__()
        self.kernel_radius = kernel_radius
        self.a0, self.a1, self.a2, self.a3 = a0, a1, a2, a3
        self.register_buffer("angle_grid", self._make_angle_grid(kernel_radius))

    @staticmethod
    def _make_angle_grid(r: int) -> torch.Tensor:
        yy, xx = torch.meshgrid(torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij")
        return torch.atan2(yy.float(), xx.float() + 1e-8)

    def forward(
        self,
        prev_fire: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        slope: torch.Tensor,
        aspect: torch.Tensor,
        temp: torch.Tensor,
        rh: torch.Tensor,
        ndvi: torch.Tensor,
        barrier: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, _, H, W = prev_fire.shape
        wind_angle = torch.atan2(v, u + 1e-8)
        wind_speed = torch.sqrt(u ** 2 + v ** 2)
        ws_norm = torch.clamp(wind_speed / 10.0, 0, 1)
        slope_norm = torch.clamp(slope, 0, 1)

        r = self.kernel_radius
        K = 2 * r + 1
        ang_flat = self.angle_grid.view(1, 1, K * K, 1, 1).to(prev_fire)

        wa = wind_angle.unsqueeze(2)
        asp = aspect.unsqueeze(2)
        aw = ws_norm.unsqueeze(2)
        as_ = slope_norm.unsqueeze(2)

        dtheta_w = ang_flat - wa
        dtheta_s = ang_flat - asp
        Ww = torch.exp(aw * torch.cos(dtheta_w))
        Ws = torch.exp(as_ * torch.cos(dtheta_s))
        kernel_flat = Ww * Ws
        kernel_flat = kernel_flat / (kernel_flat.sum(dim=2, keepdim=True) + 1e-8)

        ker = kernel_flat.reshape(B, K * K, H * W)
        # F.unfold currently falls back to CPU on MPS, so emulate im2col with tensor.unfold
        pf_pad = F.pad(prev_fire, (r, r, r, r))
        pf_unfold = (
            pf_pad.unfold(2, K, 1)
            .unfold(3, K, 1)
            .contiguous()
            .view(B, 1, H, W, K * K)
            .permute(0, 4, 2, 3, 1)
            .reshape(B, K * K, H * W)
        )
        spread = (pf_unfold * ker).sum(dim=1).view(B, 1, H, W)

        damp = torch.sigmoid(self.a0 + self.a1 * temp - self.a2 * rh + self.a3 * ndvi)
        spread = spread * damp
        if barrier is not None:
            spread = spread * (1.0 - barrier.clamp(0, 1))

        Wx = torch.cos(wind_angle)
        Wy = torch.sin(wind_angle)
        return torch.cat([spread, Wx, Wy, damp], dim=1)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsUNet(nn.Module):
    def __init__(self, in_ch: int = 16, out_ch: int = 1, base: int = 80) -> None:
        super().__init__()
        self.down1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottom = DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.conv3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.conv2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.conv1 = DoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        xb = self.bottom(self.pool3(x3))
        x = self.up3(xb)
        x = self.conv3(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, x2], dim=1))
        x = self.up1(x)
        x = self.conv1(torch.cat([x, x1], dim=1))
        return self.outc(x)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if not v.dtype.is_floating_point:
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=False)


class PolyakAverager:
    def __init__(self, model: nn.Module) -> None:
        self.count = 0
        self.shadow = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.count += 1
        for k, v in model.state_dict().items():
            if not v.dtype.is_floating_point:
                self.shadow[k].copy_(v)
            else:
                self.shadow[k].add_(v.detach() - self.shadow[k], alpha=1.0 / self.count)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=False)


class PhysicsFeatureBuilder:
    """Converts raw 9-channel tensors into the 16-channel physics-enhanced input."""

    def __init__(
        self,
        channel_names: List[str],
        mean: torch.Tensor,
        std: torch.Tensor,
        prior: PhysicsPrior,
        skip_norm: Optional[Set[str]] = None,
    ) -> None:
        self.channel_to_idx = {name: idx for idx, name in enumerate(channel_names)}
        self.available_channels = set(self.channel_to_idx)
        self.prior = prior
        self.skip_norm = set(skip_norm or {"prev_fire", "barrier"})

        mean_adj = mean.clone().detach()
        std_adj = std.clone().detach()
        for name in self.skip_norm:
            if name in self.channel_to_idx:
                idx = self.channel_to_idx[name]
                mean_adj[idx] = 0.0
                std_adj[idx] = 1.0
        self.registered_mean = mean_adj.view(1, -1, 1, 1)
        self.registered_std = (std_adj.view(1, -1, 1, 1).clamp(min=1e-6))

    def _select_optional(self, x: torch.Tensor, name: str, fill_value: float = 0.0) -> torch.Tensor:
        if name in self.channel_to_idx:
            idx = self.channel_to_idx[name]
            return x[:, idx:idx + 1]
        shape = (x.shape[0], 1, x.shape[2], x.shape[3])
        return x.new_full(shape, fill_value)

    def _gather_base(self, x: torch.Tensor) -> torch.Tensor:
        pieces = []
        for name in PHYSICS_BASE_CHANNELS:
            if name in self.channel_to_idx:
                idx = self.channel_to_idx[name]
                pieces.append(x[:, idx:idx + 1])
            else:
                shape = (x.shape[0], 1, x.shape[2], x.shape[3])
                pieces.append(x.new_zeros(shape))
        return torch.cat(pieces, dim=1)

    def __call__(self, X_raw: torch.Tensor) -> torch.Tensor:
        mean = self.registered_mean.to(X_raw.device, dtype=X_raw.dtype)
        std = self.registered_std.to(X_raw.device, dtype=X_raw.dtype)
        X_norm = (X_raw - mean) / std

        pf = self._select_optional(X_raw, "prev_fire")
        u = self._select_optional(X_raw, "u")
        v = self._select_optional(X_raw, "v")
        temp = self._select_optional(X_raw, "temp")
        rh = self._select_optional(X_raw, "rh")
        ndvi = self._select_optional(X_raw, "ndvi")
        slope = self._select_optional(X_raw, "slope")
        aspect = self._select_optional(X_raw, "aspect")
        barrier = self._select_optional(X_raw, "barrier")

        asp_cos = torch.cos(aspect)
        asp_sin = torch.sin(aspect)
        ws = torch.clamp(torch.sqrt(u ** 2 + v ** 2) / 10.0, 0, 1)

        with torch.no_grad():
            phys = self.prior(pf, u, v, slope, aspect, temp, rh, ndvi, barrier)

        base_stack = self._gather_base(X_norm)
        return torch.cat([base_stack, asp_cos, asp_sin, ws, phys], dim=1)

    @property
    def output_channels(self) -> int:
        return len(PHYSICS_BASE_CHANNELS) + 3 + 4


def build_physics_unet_bundle(CHANNELS_FOR_MODEL, meanC, stdC,
                              base_width: int = 80, ema_decay: float = 0.999,
                              loss_type: str = "bce",
                              loss_kwargs: Optional[Dict[str, float]] = None,
                              allowed_missing: Optional[Set[str]] = None):
    """Helper that mirrors the EMA/Polyak setup from the notebooks, plus loss factory."""

    missing = [c for c in PHYSICS_BASE_CHANNELS if c not in CHANNELS_FOR_MODEL]
    allowed_missing = set(OPTIONAL_PHYSICS_BASE_CHANNELS if allowed_missing is None else allowed_missing)
    unsupported = [c for c in missing if c not in allowed_missing]
    if unsupported:
        raise ValueError(
            f"Physics UNet requires base channels {PHYSICS_BASE_CHANNELS}; missing unsupported channels {unsupported}"
        )
    if missing:
        print(f"PhysicsPrior bundle: proceeding without channels {missing}")

    prior = PhysicsPrior(kernel_radius=4, a0=0.0, a1=0.03, a2=0.02, a3=0.7).to(device)
    feature_builder = PhysicsFeatureBuilder(CHANNELS_FOR_MODEL, meanC, stdC, prior)
    model = PhysicsUNet(in_ch=feature_builder.output_channels, out_ch=1, base=base_width).to(device)
    ema = EMA(model, decay=ema_decay)
    polyak = PolyakAverager(model)
    loss_conf = dict(loss_kwargs or {})
    criterion = build_physics_loss(loss_type, **loss_conf)

    print(
        f"PhysicsPrior UNet init -> in:{feature_builder.output_channels} base:{base_width} "
        f"| parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M"
    )
    return {
        "model": model,
        "prior": prior,
        "feature_builder": feature_builder,
        "ema": ema,
        "polyak": polyak,
        "criterion": criterion,
        "loss_config": {"type": loss_type, "kwargs": loss_conf},
    }
