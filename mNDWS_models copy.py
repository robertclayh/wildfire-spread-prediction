# containing logistic and U-Net architectures
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
    BATCH_SIZE: int = 8,
    include_channels: Optional[List[str]] = None,
    exclude_channels: Optional[Set[str]] = None,
):
    """Create datasets/loaders using full pipeline channels with easy ablation hooks."""
    channels = configure_channels(CHANNELS_FOR_MODEL, include_channels, exclude_channels)
    paths = mndws_dp.WildfirePaths(NPZ_ROOT)

    train_ds = mndws_dp.WildfireDataset(paths, split='train', max_samples=1200, channels=channels)
    val_ds   = mndws_dp.WildfireDataset(paths, split='eval',  max_samples=300,  channels=channels)
    test_ds  = mndws_dp.WildfireDataset(paths, split='test',  max_samples=300,  channels=channels)

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


def build_physics_loss(loss_type: str = "bce", *, pos_weight: Optional[torch.Tensor] = None,
                       focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                       tversky_alpha: float = 0.5, tversky_beta: float = 0.5,
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
    raise ValueError(f"Unsupported loss_type '{loss_type}' (expected bce|focal|tversky)")

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


# =============================================================
# 4) Physics-prior UNet + EMA / Polyak trackers
# =============================================================

PHYSICS_BASE_CHANNELS = [
    "prev_fire", "u", "v", "temp", "rh", "ndvi", "slope", "aspect", "barrier"
]


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
        pf_unfold = F.unfold(prev_fire, kernel_size=K, padding=r)
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
        self.prior = prior
        self.base_indices = [self.channel_to_idx[c] for c in PHYSICS_BASE_CHANNELS]
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

    def _select(self, x: torch.Tensor, name: str) -> torch.Tensor:
        idx = self.channel_to_idx[name]
        return x[:, idx:idx + 1]

    def _gather_base(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x[:, idx:idx + 1] for idx in self.base_indices], dim=1)

    def __call__(self, X_raw: torch.Tensor) -> torch.Tensor:
        mean = self.registered_mean.to(X_raw.device, dtype=X_raw.dtype)
        std = self.registered_std.to(X_raw.device, dtype=X_raw.dtype)
        X_norm = (X_raw - mean) / std

        pf = self._select(X_raw, "prev_fire")
        u = self._select(X_raw, "u")
        v = self._select(X_raw, "v")
        temp = self._select(X_raw, "temp")
        rh = self._select(X_raw, "rh")
        ndvi = self._select(X_raw, "ndvi")
        slope = self._select(X_raw, "slope")
        aspect = self._select(X_raw, "aspect")
        barrier = self._select(X_raw, "barrier") if "barrier" in self.channel_to_idx else None

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
                              loss_kwargs: Optional[Dict[str, float]] = None):
    """Helper that mirrors the EMA/Polyak setup from the notebooks, plus loss factory."""

    missing = [c for c in PHYSICS_BASE_CHANNELS if c not in CHANNELS_FOR_MODEL]
    if missing:
        raise ValueError(f"Physics UNet requires base channels {PHYSICS_BASE_CHANNELS}; missing {missing}")

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
