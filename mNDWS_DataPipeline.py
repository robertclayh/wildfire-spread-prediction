#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =========================================================
# 0) Setup (Colab installs) + Utilities
# =========================================================
# get_ipython().system('pip -q install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121')
# get_ipython().system('pip -q install numpy pandas scikit-learn einops tqdm')
# get_ipython().system('pip -q install kagglehub tensorflow')

import os, math, random, glob, sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import average_precision_score, precision_recall_curve

_suppress_init_logs = os.environ.get("MNDWS_PIPELINE_SILENT", "").lower() in {"1","true","yes","y"}
_already_logged_init = getattr(sys.modules[__name__], "_MNDWS_PIPELINE_LOGGED", False)
_should_log_init = (not _suppress_init_logs) and (not _already_logged_init)
sys.modules[__name__]._MNDWS_PIPELINE_LOGGED = True

def _log_init(msg: str):
    if _should_log_init:
        print(msg)

def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
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
_log_init(f"Device: {device}")


# In[2]:


# =========================================================
# 1) Data: use existing NPZ tiles; if missing, build from Kaggle TFRecords (mNDWS)
# =========================================================
import os, glob, math, getpass
import numpy as np
from tqdm import tqdm

def pick_npz_root(subdir="wildfire_npz_tiles_mndws_v1"):
    user = os.environ.get("USER") or getpass.getuser() or "user"
    candidates = [
        os.environ.get("NPZ_ROOT"),                                   # explicit override (full path)
        os.path.join(os.environ["SCRATCH"], subdir) if "SCRATCH" in os.environ else None,
        f"/scratch/{user}/{subdir}",                                   # Rivanna default
        f"/content/{subdir}" if os.path.isdir("/content") else None,   # Colab
        os.path.join(os.path.expanduser("~"), subdir),                 # fallback
    ]
    for p in candidates:
        if not p: 
            continue
        try:
            os.makedirs(p, exist_ok=True)
            return p
        except OSError:
            continue
    raise RuntimeError("Could not create NPZ root in any candidate location")

NPZ_ROOT = pick_npz_root()
_log_init(f"NPZ_ROOT -> {NPZ_ROOT}")

def have_npz(root):
    return len(glob.glob(os.path.join(root, "*.npz"))) > 0

if not have_npz(NPZ_ROOT):
    print("No NPZ tiles found — converting from mNDWS TFRecords...")
    import kagglehub, tensorflow as tf

    # Modified Next Day Wildfire Spread dataset
    path = kagglehub.dataset_download("georgehulsey/modified-next-day-wildfire-spread")
    print("Kaggle dataset path:", path)

    # Look recursively in case TFRecords are split by split folders
    tfrecs = sorted(
        glob.glob(os.path.join(path, "**", "*.tfrecord"), recursive=True)
        + glob.glob(os.path.join(path, "**", "*.tfrecords"), recursive=True)
    )
    assert len(tfrecs) > 0, "No TFRecords found in mNDWS dataset."

    os.makedirs(NPZ_ROOT, exist_ok=True)

    # mNDWS expected feature keys (each 64x64 flattened to 4096)
    # Include viirs_* masks + legacy aliases for robustness.
    keys = [
        # labels/masks
        "viirs_PrevFireMask","viirs_FireMask",
        "PrevFireMask","FireMask",
        # vegetation/topography
        "NDVI","elevation",
        # population for barrier proxy
        "population",
        # mNDWS meteorology
        "avg_sph","tmp_day","tmp_75",
        "wind_avg","wdir_wind","gust_med","wdir_gust","wind_75",
        # hydrology/landcover
        "water","impervious",
        # drought and precip
        "pdsi","pr",
        # fire danger and embeddings
        "erc","bi","chili","fuel1","fuel2","fuel3",
        # optional id
        "sample_id"
    ]

    def read_flat_float32(feat):
        fl = feat.float_list.value
        if len(fl) == 0: return None
        arr = np.asarray(fl, dtype=np.float32)
        if arr.size == 4096:
            return arr.reshape(64, 64)
        s = int(round(math.sqrt(arr.size)))
        assert s*s == arr.size, f"Unexpected length {arr.size}"
        return arr.reshape(s, s)

    def wind_uv_by(speed, theta):
        if speed is None or theta is None:
            return None, None
        th = theta.copy()
        # mNDWS dirs commonly in radians ([-pi, pi]); convert if degrees.
        if np.nanmax(np.abs(th)) > 6.4:
            th = np.deg2rad(th % 360.0)
        u = speed * np.cos(th)
        v = speed * np.sin(th)
        return u.astype(np.float32), v.astype(np.float32)

    def slope_aspect_from_elevation(z):
        gy, gx = np.gradient(z.astype(np.float32))
        mag = np.sqrt(gx**2 + gy**2)
        q95 = np.percentile(mag, 95) + 1e-6
        slope  = np.clip(mag / q95, 0, 1).astype(np.float32)
        aspect = np.arctan2(-gy, -gx).astype(np.float32)  # [-pi, pi]
        return slope, aspect

    def ndvi_to_01(ndvi):
        nd = ndvi.astype(np.float32)
        # mNDWS NDVI is scaled (e.g., [-2000..9987]). Convert to [-1,1] via /10000
        if np.nanmax(np.abs(nd)) > 2.0:
            nd = nd / 10000.0
        # Map [-1,1] -> [0,1]
        nd = np.clip((nd + 1.0) / 2.0, 0, 1)
        return nd.astype(np.float32)

    def rh_from_avg_sph(avg_sph, tmp_day, tmp_75):
        # Proxy RH: normalize specific humidity by its 95th percentile
        # and damp by diurnal range signal (tmp_75 - tmp_day).
        if avg_sph is None:
            return np.zeros((64,64), np.float32)
        s95 = np.percentile(avg_sph, 95) + 1e-6
        rh = np.clip(avg_sph / s95, 0, 1)
        if tmp_day is not None and tmp_75 is not None:
            tr = np.clip((tmp_75 - tmp_day), 0, 20) / 20.0
            rh = rh * (1.0 - 0.5*tr)
        return np.clip(rh, 0, 1).astype(np.float32)

    def barrier_from_population(pop):
        if pop is None:
            return np.zeros((64,64), np.float32)
        pop = np.clip(pop, 0, None).astype(np.float32)
        thr = np.percentile(pop, 90)
        return (pop >= thr).astype(np.float32)

    def pick_first(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    converted = 0
    skipped_missing_masks = 0

    for f in tqdm(tfrecs, desc="Converting TFRecords → NPZ (mNDWS)"):
        for raw in tf.data.TFRecordDataset(f):
            ex = tf.train.Example.FromString(raw.numpy()).features.feature
            A = {k: read_flat_float32(ex[k]) if k in ex else None for k in keys}

            # Masks: prefer viirs_*; fall back to legacy names
            prev_mask = pick_first(A.get("viirs_PrevFireMask"), A.get("PrevFireMask"))
            next_mask = pick_first(A.get("viirs_FireMask"),     A.get("FireMask"))
            if prev_mask is None or next_mask is None:
                skipped_missing_masks += 1
                continue

            # Labels
            prev_fire = (prev_mask > 0.5).astype(np.float32)
            next_fire = (next_mask > 0.5).astype(np.float32)

            # Temperature: use mNDWS daily mean (tmp_day)
            if A["tmp_day"] is not None:
                temp = A["tmp_day"].astype(np.float32)
            else:
                temp = np.zeros((64,64), np.float32)

            # Wind: use wind_avg + wdir_wind (mNDWS)
            u, v = wind_uv_by(A["wind_avg"], A["wdir_wind"])
            if u is None or v is None:
                # Fallback to gust-based if average missing
                u, v = wind_uv_by(A["gust_med"], A["wdir_gust"])
                if u is None or v is None:
                    u = np.zeros((64,64), np.float32)
                    v = np.zeros((64,64), np.float32)

            # NDVI: rescale from scaled ints, then to [0,1]
            if A["NDVI"] is not None:
                ndvi = ndvi_to_01(A["NDVI"])
            else:
                ndvi = np.full((64,64), 0.5, np.float32)

            # Relative humidity proxy from avg_sph and temperatures
            rh = rh_from_avg_sph(A["avg_sph"], A["tmp_day"], A["tmp_75"])

            # Terrain
            if A["elevation"] is not None:
                slope, aspect = slope_aspect_from_elevation(A["elevation"])
            else:
                slope = np.zeros((64,64), np.float32)
                aspect = np.zeros((64,64), np.float32)

            # Barrier: population-based proxy
            barrier = barrier_from_population(A["population"])

            # Save minimal, consistent fields your pipeline expects.
            fields = dict(
                prev_fire=prev_fire, next_fire=next_fire,
                u=u, v=v, temp=temp, rh=rh, ndvi=ndvi,
                slope=slope, aspect=aspect, barrier=barrier
            )

            # Optional: keep extra mNDWS channels
            def _scale01(x, denom): return np.clip((x/denom), 0, 1).astype(np.float32)
            if A["impervious"] is not None: fields["impervious"] = _scale01(A["impervious"], 100.0)
            if A["water"] is not None:      fields["water"]      = _scale01(A["water"], 100.0)
            for k in ["erc","pdsi","pr","bi","chili","fuel1","fuel2","fuel3","wind_75","gust_med"]:
                if A.get(k) is not None: fields[k] = A[k].astype(np.float32)

            sid_feat = ex.get("sample_id", None)
            if sid_feat and len(sid_feat.bytes_list.value) > 0:
                sid = sid_feat.bytes_list.value[0].decode("utf-8")
            else:
                sid = f"{os.path.basename(f)}_{converted:07d}"

            np.savez(os.path.join(NPZ_ROOT, f"{sid}.npz"), **fields)
            converted += 1

    print(f"Converted {converted} tiles → {NPZ_ROOT}")
    print(f"Skipped (no masks): {skipped_missing_masks}")
else:
    _log_init(f"Using existing NPZ tiles at {NPZ_ROOT} (found {len(glob.glob(os.path.join(NPZ_ROOT, '*.npz')))} files)")


# In[3]:


# =========================================================
# 2) Dataset & configurable channel set (mNDWS)
# =========================================================
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from dataclasses import dataclass

# Base 9 channels used in NDWS workflows
CH_ORDER_BASE = [
    "prev_fire",    # 0
    "u",
    "v",
    "temp",
    "rh",
    "ndvi",
    "slope",
    "aspect",
    "barrier"       # 8
]

# Extra mNDWS channels saved by your converter (if present)
CH_ORDER_EXTRA = [
    "erc","pdsi","pr","bi","chili",
    "fuel1","fuel2","fuel3",
    "impervious","water",
    "wind_75","gust_med"
]

# Choose what to use
USE_CHANNELS = CH_ORDER_BASE + CH_ORDER_EXTRA  # or just CH_ORDER_BASE

# Channels that should pass through unchanged (no normalization)
DO_NOT_NORMALIZE = {"prev_fire", "barrier"}

@dataclass
class WildfirePaths:
    root: str  # NPZ_ROOT

_SPLIT_TO_ID = {"train": 0, "test": 1, "eval": 2, "val": 2}

def _infer_split_from_dir(root: str, split: str):
    sub = "eval" if split == "val" else split
    split_dir = os.path.join(root, sub)
    if os.path.isdir(split_dir):
        files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
        if files:
            return files
    return None

def _filter_by_split_id(files, split: str):
    wanted = _SPLIT_TO_ID[split]
    out = []
    for f in files:
        try:
            with np.load(f, mmap_mode="r") as z:
                if "split_id" in z and int(np.array(z["split_id"])) == wanted:
                    out.append(f)
        except Exception:
            pass
    return out

class WildfireDataset(Dataset):
    def __init__(self, paths: WildfirePaths, split="train", max_samples=None, seed=1337, channels=None):
        if split not in ("train","val","test","eval"):
            raise ValueError("split must be one of: train|val|test|eval")
        self.channels = list(USE_CHANNELS if channels is None else channels)

        # Prefer per-split subfolders, else split_id, else 70/15/15 fallback
        files = _infer_split_from_dir(paths.root, split)
        if files is None:
            all_files = sorted(glob.glob(os.path.join(paths.root, "*.npz")))
            if not all_files:
                raise ValueError(f"No .npz files under {paths.root}")
            has_split_id = False
            for probe in all_files[:10]:
                try:
                    with np.load(probe, mmap_mode="r") as z:
                        if "split_id" in z:
                            has_split_id = True
                            break
                except Exception:
                    continue
            if has_split_id:
                files = _filter_by_split_id(all_files, split)
            else:
                rng = np.random.default_rng(seed)
                shuffled = all_files.copy()
                rng.shuffle(shuffled)
                n = len(shuffled)
                n_train = int(round(0.70 * n))
                n_val = int(round(0.15 * n))
                if split == "train":
                    sel = np.arange(0, max(1, n_train))
                elif split in ("val","eval"):
                    sel = np.arange(n_train, max(n_train+1, n_train+n_val))
                else:
                    sel = np.arange(n_train+n_val, n) if (n_train+n_val) < n else np.arange(n-1, n)
                files = [shuffled[i] for i in sel]

        if max_samples:
            files = files[:max_samples]
        if len(files) == 0:
            any_file = sorted(glob.glob(os.path.join(paths.root, "**", "*.npz"), recursive=True))
            if not any_file:
                raise ValueError(f"No .npz files under {paths.root}")
            files = [any_file[0]]

        self.paths = paths
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        arr = np.load(self.files[i])
        # Always require minimal fields for labeling and core physics
        required = {"prev_fire","next_fire","u","v","temp","rh","ndvi","slope","aspect"}
        missing = [k for k in required if k not in arr]
        if missing:
            raise KeyError(f"{os.path.basename(self.files[i])} missing keys: {missing}")

        # Build X_raw in configured order; fill missing optional channels with zeros
        chans = []
        for k in self.channels:
            if k not in arr:
                # Gracefully handle optional extras missing in some NPZs
                chans.append(np.zeros_like(arr["prev_fire"], dtype=np.float32)[None, ...])
            else:
                chans.append(arr[k][None, ...].astype(np.float32))
        X_raw = np.concatenate(chans, axis=0)

        y = arr["next_fire"][None, ...].astype(np.float32)
        return {"X_raw": torch.from_numpy(X_raw), "y": torch.from_numpy(y)}

# Build datasets/loaders
paths = WildfirePaths(NPZ_ROOT)
train_ds = WildfireDataset(paths, split="train", max_samples=1200)
val_ds   = WildfireDataset(paths, split="eval",  max_samples=300)
test_ds  = WildfireDataset(paths, split="test",  max_samples=300)

def make_loader(ds, batch_size=16, upweight_positive=False, shuffle=False):
    if upweight_positive:
        weights = []
        for f in ds.files:
            try:
                y = np.load(f, mmap_mode="r")["next_fire"]
                weights.append(5.0 if y.sum() > 0 else 1.0)
            except Exception:
                weights.append(1.0)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=use_cuda,
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=use_cuda,
        persistent_workers=True,
        prefetch_factor=2,
    )

train_loader = make_loader(train_ds, batch_size=16, upweight_positive=True)
val_loader   = make_loader(val_ds,   batch_size=16)
test_loader  = make_loader(test_ds,  batch_size=16)


# In[4]:


# =========================================================
# 3) Channel Stats for selected channels
# =========================================================
import numpy as np
import torch
from torch.utils.data import DataLoader

def build_channel_index(ds):
    # Map channel name -> index for DO_NOT_NORMALIZE handling
    name_to_idx = {name: idx for idx, name in enumerate(ds.channels)}
    return name_to_idx

@torch.no_grad()
def compute_channel_stats(ds, n_max_samples=None, batch_size=32):
    C = len(ds.channels)
    sums = np.zeros(C, dtype=np.float64)
    sqs  = np.zeros(C, dtype=np.float64)
    count = np.zeros(C, dtype=np.float64)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    seen = 0
    for b in loader:
        x = b["X_raw"].numpy()          # (B,C,H,W)
        B, Cb, H, W = x.shape
        assert Cb == C
        x = x.reshape(B, C, -1)
        mask = ~np.isnan(x)
        sums += np.where(mask, x, 0.0).sum(axis=(0, 2))
        sqs  += np.where(mask, x*x, 0.0).sum(axis=(0, 2))
        count += mask.sum(axis=(0, 2))
        seen += B
        if n_max_samples and seen >= n_max_samples:
            break

    count = np.maximum(count, 1.0)
    mean = sums / count
    var  = np.maximum(sqs / count - mean**2, 1e-8)
    std  = np.sqrt(var)

    # Do not normalize certain channels
    idx = build_channel_index(ds)
    for name in DO_NOT_NORMALIZE:
        if name in idx:
            mean[idx[name]] = 0.0
            std[idx[name]]  = 1.0

    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)

meanC, stdC = compute_channel_stats(train_ds, n_max_samples=2000, batch_size=32)
meanC, stdC = meanC.to(device), stdC.to(device)
meanC.shape, stdC.shape, train_ds.channels
