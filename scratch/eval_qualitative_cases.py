import os
import numpy as np
import matplotlib.pyplot as plt
import torch

import mNDWS_models as mndws_models

# ============================================================
# 0) Setup: seed, device, data pipeline
# ============================================================
mndws_models.set_seed(1337)
device = mndws_models.device
use_cuda = mndws_models.use_cuda
use_mps = mndws_models.use_mps
print("Device:", device)

CHANNELS_FOR_MODEL = mndws_models.configure_channels()
train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, meanC, stdC = mndws_models.pipeline_hookup(
    CHANNELS_FOR_MODEL=CHANNELS_FOR_MODEL,
    BATCH_SIZE=16,
)

def build_lr_input(X_raw0, mean=None, std=None):
    mean_t = mean if mean is not None else meanC
    std_t = std if std is not None else stdC
    return mndws_models.build_lr_input(X_raw0, mean_t, std_t)

print(f"Channels configured ({len(CHANNELS_FOR_MODEL)}): {CHANNELS_FOR_MODEL}")


# ============================================================
# 1) Load LR baseline from artifact
# ============================================================
LR_ART_PATH = os.path.join(
    os.getcwd(), "pixel_logreg.pt")

print("LR artifact path:   ", LR_ART_PATH)
artifact_lr = torch.load(LR_ART_PATH, map_location=device)

in_ch_lr = artifact_lr["model"]["in_ch"]
lr_model = mndws_models.PixelLogReg(in_ch=in_ch_lr).to(device)
lr_model.load_state_dict(artifact_lr["state_dict"])

LR_THR = float(artifact_lr["best_thr"])
LR_MEAN = artifact_lr["mean"].to(device)
LR_STD = artifact_lr["std"].to(device)
print(f"[LR] loaded. Best threshold = {LR_THR:.3f}")


# ============================================================
# 2) Load Physics-Prior U-Net from artifact + feature builder
# ============================================================
PHYS_ART_PATH = os.path.join(os.getcwd(), "physics_unet.pt")

print("UNet artifact path: ", PHYS_ART_PATH)
artifact_unet = torch.load(PHYS_ART_PATH, map_location=device)

unet_in_ch = artifact_unet["model"]["in_ch"]
unet_base = artifact_unet["model"]["base"]

pos_weight = mndws_models.pos_weight_from_loader(train_loader)

bundle = mndws_models.build_physics_unet_bundle(
    CHANNELS_FOR_MODEL,
    meanC,
    stdC,
    base_width=unet_base,
    ema_decay=0.999,
    loss_type="hybrid",
    loss_kwargs={
        "pos_weight": pos_weight,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "focal_weight": 0.5,
        "tversky_alpha": 0.7,
        "tversky_beta": 0.3,
    },
)

physics_model = bundle["model"].to(device)
feature_builder = bundle["feature_builder"]
physics_model.load_state_dict(artifact_unet["state_dict"])

UNET_THR = float(artifact_unet["best_thr"])
print(f"[PhysicsPrior U-Net] loaded. Best threshold = {UNET_THR:.3f}")


# ============================================================
# 3) Per-tile prediction helper
# ============================================================
@torch.no_grad()
def get_tile_predictions(idx, ds,
                         lr_model, unet_model,
                         lr_thr=LR_THR, unet_thr=UNET_THR,
                         prev_channel_name="prev_fire"):
    sample = ds[idx]
    X_raw = sample["X_raw"].unsqueeze(0).to(device)   # (1,C,H,W)
    y = sample["y"][0].cpu().numpy()                  # (H,W)

    channel_names = list(ds.channels)
    prev_idx = channel_names.index(prev_channel_name) if prev_channel_name in channel_names else 0
    prev_fire = sample["X_raw"][prev_idx].cpu().numpy()

    # LR prediction
    X_lr = build_lr_input(X_raw, mean=LR_MEAN, std=LR_STD)
    p_lr = torch.sigmoid(lr_model(X_lr))[0, 0].cpu().numpy()
    lr_mask = (p_lr >= lr_thr).astype(np.float32)

    # Physics-Prior U-Net prediction
    feats = feature_builder(X_raw)
    p_unet = torch.sigmoid(unet_model(feats))[0, 0].cpu().numpy()
    unet_mask = (p_unet >= unet_thr).astype(np.float32)

    return prev_fire, lr_mask, unet_mask, y


# ============================================================
# 4) Single-case plotting helper (1 row, 4 columns)
# ============================================================
@torch.no_grad()
def plot_single_case(idx, ds,
                     lr_model, unet_model,
                     lr_thr=LR_THR, unet_thr=UNET_THR,
                     prev_channel_name="prev_fire",
                     title_prefix="",
                     save_path=None):
    prev_fire, lr_mask, unet_mask, gt = get_tile_predictions(
        idx, ds, lr_model, unet_model,
        lr_thr=lr_thr, unet_thr=unet_thr,
        prev_channel_name=prev_channel_name,
    )

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    axes[0].imshow(prev_fire, vmin=0, vmax=1, cmap="gray")
    axes[0].set_title(f"{title_prefix}Prev burn (idx={idx})")
    axes[0].axis("off")

    axes[1].imshow(lr_mask, vmin=0, vmax=1, cmap="hot")
    axes[1].set_title("LR prediction")
    axes[1].axis("off")

    axes[2].imshow(unet_mask, vmin=0, vmax=1, cmap="hot")
    axes[2].set_title("Physics-Prior U-Net")
    axes[2].axis("off")

    axes[3].imshow(gt, vmin=0, vmax=1, cmap="hot")
    axes[3].set_title("Next-day truth")
    axes[3].axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure → {save_path}")
    else:
        plt.show()
    plt.close(fig)

# ============================================================
# 5) Compute per-tile IoU for LR and Physics-Prior U-Net
# ============================================================
@torch.no_grad()
def compute_iou_per_tile(ds, lr_model, unet_model,
                         lr_thr=LR_THR, unet_thr=UNET_THR):
    n = len(ds)
    iou_lr = np.zeros(n, dtype=np.float32)
    iou_unet = np.zeros(n, dtype=np.float32)
    gt_area = np.zeros(n, dtype=np.float32)

    for i in range(n):
        _, lr_mask, unet_mask, gt = get_tile_predictions(
            i, ds, lr_model, unet_model,
            lr_thr=lr_thr, unet_thr=unet_thr,
        )
        gt_bin = (gt > 0.5).astype(np.uint8)
        gt_area[i] = gt_bin.sum()

        if gt_area[i] == 0:
            # no fire in truth → IoU defined as 0 for our selection purposes
            continue

        # IoU for LR
        inter_lr = np.logical_and(lr_mask == 1, gt_bin == 1).sum()
        union_lr = np.logical_or(lr_mask == 1, gt_bin == 1).sum()
        iou_lr[i] = inter_lr / (union_lr + 1e-8)

        # IoU for U-Net
        inter_unet = np.logical_and(unet_mask == 1, gt_bin == 1).sum()
        union_unet = np.logical_or(unet_mask == 1, gt_bin == 1).sum()
        iou_unet[i] = inter_unet / (union_unet + 1e-8)

    return iou_lr, iou_unet, gt_area

# Compute tile-level metrics on the TEST split
iou_lr, iou_unet, gt_area = compute_iou_per_tile(test_ds, lr_model, physics_model,
                                                 lr_thr=LR_THR, unet_thr=UNET_THR)

# Helper to pick an index given a boolean mask and a score
def _pick_idx(mask, score_vec, description):
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        print(f"[WARN] No tiles matched criteria for: {description}")
        return None
    best_local = idxs[np.argmax(score_vec[idxs])]
    print(f"[SELECT] {description}: idx={best_local}")
    return int(best_local)

# 1) Simple success: both good IoU
success_mask = (iou_lr >= 0.45) & (iou_unet >= 0.55) & (gt_area > 0)
idx_success = _pick_idx(success_mask, (iou_lr + iou_unet) / 2,
                        "simple success (both good)")

# 2) LR failure, U-Net success: big performance gap, non-trivial fire
gap_mask = (
    (iou_lr <= 0.20) &          # LR doing poorly
    (iou_unet >= 0.45) &        # U-Net doing well
    (gt_area >= 50)             # Require some minimum fire area for visibility
)
idx_lr_fail = _pick_idx(
    gap_mask,
    (iou_unet - iou_lr),
    "LR failure, U-Net success (strict)"
)

# Fallback: if strict criteria found nothing, relax the conditions
if idx_lr_fail is None:
    fallback_mask = (gt_area >= 50) & (iou_unet > iou_lr)
    idx_lr_fail = _pick_idx(
        fallback_mask,
        (iou_unet - iou_lr),
        "LR failure, U-Net success (relaxed)"
    )

# 3) Shared failure: both poor, but nontrivial fire area
shared_mask = (iou_lr <= 0.20) & (iou_unet <= 0.20) & (gt_area > 0)
idx_shared = _pick_idx(shared_mask, gt_area,
                       "shared failure (both poor)")

if idx_success is not None:
    print("idx_success:", idx_success,
          "IoU_LR:", iou_lr[idx_success],
          "IoU_UNet:", iou_unet[idx_success],
          "area:", gt_area[idx_success])
else:
    print("idx_success: None")

if idx_lr_fail is not None:
    print("idx_lr_fail:", idx_lr_fail,
          "IoU_LR:", iou_lr[idx_lr_fail],
          "IoU_UNet:", iou_unet[idx_lr_fail],
          "area:", gt_area[idx_lr_fail])
else:
    print("idx_lr_fail: None (no suitable LR-fail / U-Net-success tile found)")

if idx_shared is not None:
    print("idx_shared:", idx_shared,
          "IoU_LR:", iou_lr[idx_shared],
          "IoU_UNet:", iou_unet[idx_shared],
          "area:", gt_area[idx_shared])
else:
    print("idx_shared: None")

if idx_success is not None:
    plot_single_case(idx_success, test_ds, lr_model, physics_model,
                     lr_thr=LR_THR, unet_thr=UNET_THR,
                     prev_channel_name="prev_fire",
                     title_prefix="Simple success – ")

if idx_lr_fail is not None:
    plot_single_case(idx_lr_fail, test_ds, lr_model, physics_model,
                     lr_thr=LR_THR, unet_thr=UNET_THR,
                     prev_channel_name="prev_fire",
                     title_prefix="LR failure, U-Net success – ")

if idx_shared is not None:
    plot_single_case(idx_shared, test_ds, lr_model, physics_model,
                     lr_thr=LR_THR, unet_thr=UNET_THR,
                     prev_channel_name="prev_fire",
                     title_prefix="Shared failure – ")



# ============================================================
# 6) Define qualitative cases and generate separate figures
# ============================================================
CASES = {
    "fig6a_simple_success.png": {
        "idx": idx_success,
        "title": "Fig. 6a: Simple success – ",
    },
    "fig6c_shared_failure.png": {
        "idx": idx_shared,
        "title": "Fig. 6c: Shared failure – ",
    },
}

if idx_lr_fail is not None:
    CASES["fig6b_lr_fail_unet_success.png"] = {
        "idx": idx_lr_fail,
        "title": "Fig. 6b: LR failure, U-Net success – ",
    }


out_dir = "./figs_qualitative"
os.makedirs(out_dir, exist_ok=True)

for fname, cfg in CASES.items():
    idx = cfg["idx"]
    title_prefix = cfg.get("title", "")
    save_path = os.path.join(out_dir, fname)
    print(f"Rendering {fname} (tile idx={idx})")
    plot_single_case(
        idx,
        test_ds,
        lr_model,
        physics_model,
        lr_thr=LR_THR,
        unet_thr=UNET_THR,
        prev_channel_name="prev_fire",
        title_prefix=title_prefix,
        save_path=save_path,
    )

print("Done generating qualitative case figures.")
