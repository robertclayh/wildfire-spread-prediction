# =========================================================
#   eval_models.py — Unified Evaluation: LR + Physics UNet
# =========================================================

import argparse, yaml, torch, numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve
import time

import mNDWS_models as mndws_models

device = mndws_models.device
use_cuda = mndws_models.use_cuda
use_mps = mndws_models.use_mps


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_build_lr_input(meanC, stdC):
    def _build(X_raw):
        return mndws_models.build_lr_input(X_raw, meanC, stdC)
    return _build



# =========================================================
#  Logistic Regression Evaluation
# =========================================================
@torch.no_grad()
def eval_lr(loader, *, model, build_lr_input, desc=None):
    model.eval()
    all_p, all_t = [], []

    for b in tqdm(loader, desc=desc or "eval LR", leave=False):
        X_raw0 = b["X_raw"].to(device, non_blocking=True)
        y = b["y"].to(device, non_blocking=True)
        X = build_lr_input(X_raw0)

        p = torch.sigmoid(model(X)).flatten().cpu().numpy()
        t = y.flatten().cpu().numpy()

        all_p.append(p)
        all_t.append(t)

    p = np.concatenate(all_p)
    t = np.concatenate(all_t)

    if t.sum() == 0:
        return 0., 0., 0.5, 0.

    ap = average_precision_score(t, p)
    prec, rec, thr = precision_recall_curve(t, p)
    f1 = (2 * prec * rec) / (prec + rec + 1e-8)
    best_idx = f1.argmax()
    best_thr = thr[best_idx] if best_idx < len(thr) else 0.5

    yhat = (p >= best_thr).astype(np.float32)
    intersection = float((yhat * t).sum())
    union = float(yhat.sum() + t.sum() - intersection)
    iou = intersection / (union + 1e-8)

    return float(ap), float(f1.max()), float(best_thr), float(iou)



# =========================================================
#  Physics UNet Evaluation
# =========================================================
@torch.no_grad()
def eval_physics(model_obj, forward_batch_fn, loader, desc="eval Physics"):
    model_obj.eval()
    all_p, all_t = [], []

    for batch in tqdm(loader, desc=desc, leave=False):
        feats, y = forward_batch_fn(batch)
        logits = model_obj(feats)

        p = torch.sigmoid(logits).flatten().cpu().numpy()
        t = y.flatten().cpu().numpy()

        all_p.append(p)
        all_t.append(t)

    p = np.concatenate(all_p)
    t = np.concatenate(all_t)

    if t.sum() == 0:
        return 0., 0., 0.5, 0.

    ap = average_precision_score(t, p)
    prec, rec, thr = precision_recall_curve(t, p)
    f1 = (2 * prec * rec) / (prec + rec + 1e-8)
    best_idx = f1.argmax()
    best_thr = thr[best_idx] if best_idx < len(thr) else 0.5

    yhat = (p >= best_thr).astype(np.float32)
    intersection = float((yhat * t).sum())
    union = float(yhat.sum() + t.sum() - intersection)
    iou = intersection / (union + 1e-8)

    return float(ap), float(f1.max()), float(best_thr), float(iou)



# =========================================================
#  LR Results Runner
# =========================================================
def lr_results(args):

    cfg = load_cfg(args.config)

    # Dataset
    channels = cfg["data"]["channels_used"]
    if channels == "auto":
        channels = None

    (train_ds, val_ds, test_ds,
     train_loader, val_loader, test_loader,
     meanC, stdC) = mndws_models.pipeline_hookup(
        CHANNELS_FOR_MODEL=channels,
        BATCH_SIZE=cfg["data"]["batch_size"]
    )

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)

    # Rebuild LR model exactly as training
    lr_model, pw, criterion, optimizer = mndws_models.PixelLogReg_outputs(
        train_ds=train_ds,
        meanC=meanC,
        stdC=stdC,
        train_loader=train_loader,
        device=device,
    )
    lr_model.load_state_dict(ckpt["lrmodel"])

    build_lr_input = make_build_lr_input(ckpt["meanC"], ckpt["stdC"])

    param_count = int(sum(p.numel() for p in lr_model.parameters() if p.requires_grad))

    # --------------- latency ----------------
    @torch.no_grad()
    def measure_latency(ds, repeats=50):
        if len(ds) == 0:
            return None
        sample = ds[0]["X_raw"].unsqueeze(0).to(device)
        X = build_lr_input(sample)

        if use_cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(repeats):
            torch.sigmoid(lr_model(X))

        if use_cuda: torch.cuda.synchronize()
        return (time.perf_counter() - t0) / repeats

    latency_s = measure_latency(test_ds, repeats=100)
    peak_gpu = (torch.cuda.max_memory_allocated(device) / (1024**3)) if use_cuda else None

    # --------------- metrics ----------------
    print("\nRunning validation metrics…")
    val_ap, val_f1, val_thr, val_iou = eval_lr(val_loader, model=lr_model, build_lr_input=build_lr_input, desc="VAL LR")

    print("Running test metrics…")
    test_ap, test_f1, test_thr, test_iou = eval_lr(test_loader, model=lr_model, build_lr_input=build_lr_input, desc="TEST LR")

    print("\n============================")
    print("   Pixel LogReg Evaluation")
    print("============================")
    print(f"Parameters           : {param_count}")
    if latency_s:  print(f"Latency (1 tile)     : {latency_s*1e3:.3f} ms")
    if peak_gpu:    print(f"Peak GPU memory      : {peak_gpu:.3f} GB")

    print("\nValidation:")
    print(f"  AP={val_ap:.4f}, F1={val_f1:.4f}, IoU={val_iou:.4f}, thr≈{val_thr:.3f}")

    print("\nTest:")
    print(f"  AP={test_ap:.4f}, F1={test_f1:.4f}, IoU={test_iou:.4f}, thr≈{test_thr:.3f}")



# =========================================================
#  UNet Results Runner
# =========================================================
def unet_results(args):
    # Define _forward_batch locally for UNet evaluation, using the current feature_builder
    def _forward_batch(model_obj, batch):
        X_raw = batch["X_raw"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        feats = feature_builder(X_raw) # Use feature_builder from unet_results scope
        return feats, y

    cfg = load_cfg(args.config)

    channels = cfg["data"]["channels_used"]
    if channels == "auto":
        channels = None

    (train_ds, val_ds, test_ds,
     train_loader, val_loader, test_loader,
     meanC, stdC) = mndws_models.pipeline_hookup(
        CHANNELS_FOR_MODEL=channels,
        BATCH_SIZE=cfg["data"]["batch_size"]
    )

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    # ALWAYS use dataset channels — not cfg
    model_channels = train_ds.channels

    # Rebuild bundle (same as training)
    bundle = mndws_models.build_physics_unet_bundle(
        model_channels,
        meanC,
        stdC,
        base_width=80,
        ema_decay=0.0,
        loss_type="hybrid",
        loss_kwargs={},
    )

    feature_builder = bundle["feature_builder"]
    # Local helper for UNet evaluation to capture feature_builder
    def _forward_batch_unet(batch):
        X_raw = batch["X_raw"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        feats = feature_builder(X_raw)
        return feats, y

    in_channels = feature_builder.output_channels

    physics_model = mndws_models.PhysicsUNet(
        in_ch=in_channels,
        out_ch=1,
        base=80,
    ).to(device)

    if "physicsmodel" in ckpt:
        physics_model.load_state_dict(ckpt["physicsmodel"])
    else:
        physics_model.load_state_dict(ckpt["model"])

    # Print compute metrics summary
    print("\n[Physics] Compute metrics available:")
    for k, v in ckpt.get("compute_metrics", {}).items():
        print(f"  {k:28s} : {v}")

    # Build variants: Raw + EMA + Polyak
    variants = {"Raw": physics_model}

    if ckpt.get("ema_tracker", None) is not None:
        m = mndws_models.PhysicsUNet(in_ch=in_channels, out_ch=1, base=80).to(device)
        ckpt["ema_tracker"].copy_to(m)
        variants["EMA"] = m

    if ckpt.get("polyak_tracker", None) is not None:
        m = mndws_models.PhysicsUNet(in_ch=in_channels, out_ch=1, base=80).to(device)
        ckpt["polyak_tracker"].copy_to(m)
        variants["Polyak"] = m

    # Evaluate
    results = {}
    for name, model_obj in variants.items():
        ap_v, f1_v, thr_v, iou_v = eval_physics(model_obj, _forward_batch_unet, val_loader, desc=f"VAL {name}")
        ap_t, f1_t, thr_t, iou_t = eval_physics(model_obj, _forward_batch_unet, test_loader, desc=f"TEST {name}")

        results[name] = dict(
            val_ap=ap_v, val_f1=f1_v, val_iou=iou_v, val_thr=thr_v,
            test_ap=ap_t, test_f1=f1_t, test_iou=iou_t, test_thr=thr_t,
        )

    # Print results
    print("\nFinal metrics (val/test):")
    for name, st in results.items():
        print(
            f"  {name:6s} | VAL AP {st['val_ap']:.4f} "
            f"F1 {st['val_f1']:.4f} IoU {st['val_iou']:.4f} thr≈{st['val_thr']:.3f} | "
            f"TEST AP {st['test_ap']:.4f} F1 {st['test_f1']:.4f} IoU {st['test_iou']:.4f}"
        )

    return results



# =========================================================
#  Main: Multi-Checkpoint, Auto Model-Type Detection
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True, nargs="+")
    parser.add_argument("--model-type", choices=["lr", "unet"], default=None)
    args = parser.parse_args()

    for ck in args.ckpt:
        print("\n========================================")
        print(f" Evaluating checkpoint: {ck}")
        print("========================================")

        if args.model_type:
            mt = args.model_type
        else:
            low = ck.lower()
            if "unet" in low:
                mt = "unet"
            elif "lr" in low or "logreg" in low:
                mt = "lr"
            else:
                print(f"Cannot infer model type for {ck}. Use --model-type.")
                continue

        sub = argparse.Namespace(config=args.config, ckpt=ck)

        if mt == "lr":
            lr_results(sub)
        else:
            unet_results(sub)

        print("========================================\n")


if __name__ == "__main__":
    main()
