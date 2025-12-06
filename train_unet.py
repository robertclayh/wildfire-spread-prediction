"""Train the PhysicsPrior UNet wildfire model.

Run this script from the repo root to reuse the shared mNDWS data pipeline,
fit the PhysicsPrior UNet with EMA/Polyak tracking, and save checkpoints plus
channel stats. Provide `--config path/to/config.yaml` for full control or rely
on the CLI defaults (epochs, batch size, output path, checkpoint resume).

Example
-------
Train for a single quick epoch using the default config fallback:

    python train_unet.py --epochs 1 --output outputs/unet_smoke_test.pt
"""

#training script for unet model
#set up 
import argparse, yaml, torch, pathlib 
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve
import os, math, random, glob
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from contextlib import nullcontext

#our packages
import mNDWS_models as mndws_models
import mNDWS_DataPipeline as mndws_dp

device = mndws_models.device
use_cuda = mndws_models.use_cuda
use_mps = mndws_models.use_mps

DEFAULT_LOSS_CFG = {
    "focal_alpha": 0.25,
    "focal_gamma": 2.0,
    "focal_weight": 0.5,
    "tversky_alpha": 0.7,
    "tversky_beta": 0.3,
}

def load_configuration(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def main():
    #configure yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to YAML config (optional)")
    parser.add_argument("--output", default="outputs/unet_final.pt", help="Path to save trained checkpoint")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs if config not provided")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size if config not provided")
    args = parser.parse_args()

    if args.config:
        cfg = load_configuration(args.config)
        EPOCHS_PHYSICS = cfg["training"]["epochs"]
        channels_cfg = cfg["data"].get("channels_used", "auto")
        batch_size = cfg["data"].get("batch_size", args.batch_size)
        loss_cfg = cfg.get("loss", DEFAULT_LOSS_CFG)
    else:
        EPOCHS_PHYSICS = args.epochs
        channels_cfg = "auto"
        batch_size = args.batch_size
        loss_cfg = DEFAULT_LOSS_CFG

    if channels_cfg == "auto":
        channels = mndws_dp.USE_CHANNELS
    else:
        channels = channels_cfg

    #data pipeline
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, meanC, stdC = mndws_models.pipeline_hookup(
    CHANNELS_FOR_MODEL=channels,
    BATCH_SIZE=batch_size,
 )
    # =========================================================
    # PhysicsPrior UNet bundle + optimizer/criterion setup
    # =========================================================
    pos_weight = mndws_models.pos_weight_from_loader(train_loader)

    bundle = mndws_models.build_physics_unet_bundle(
        channels,
        meanC,
        stdC,
        base_width=80,
        ema_decay=0.999,
        loss_type="hybrid",  # combines focal + Tversky
        loss_kwargs={
            "pos_weight": pos_weight,
            "focal_alpha": loss_cfg.get("focal_alpha", DEFAULT_LOSS_CFG["focal_alpha"]),
            "focal_gamma": loss_cfg.get("focal_gamma", DEFAULT_LOSS_CFG["focal_gamma"]),
            "focal_weight": loss_cfg.get("focal_weight", DEFAULT_LOSS_CFG["focal_weight"]),  # 0→pure Tversky, 1→pure focal
            "tversky_alpha": loss_cfg.get("tversky_alpha", DEFAULT_LOSS_CFG["tversky_alpha"]),
            "tversky_beta": loss_cfg.get("tversky_beta", DEFAULT_LOSS_CFG["tversky_beta"]),
        },
    )
    physics_model = bundle["model"]

    if args.ckpt is not None:
      if os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        physics_model.load_state_dict(ckpt['physicsmodel'])
        meanC = ckpt.get("meanC", meanC)
        stdC = ckpt.get("stdC", stdC)
      else:
        raise FileNotFoundError(f"Checkpoint file not found:")
    else:
      ckpt = None
      print("No checkpoint, training from scratch")

    feature_builder = bundle["feature_builder"]
    ema_tracker = bundle["ema"]
    polyak_tracker = bundle["polyak"]
    criterion = bundle["criterion"]

    optimizer = torch.optim.AdamW(physics_model.parameters(), lr=2e-4, weight_decay=1e-4)
    amp_enabled = use_cuda
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
    if amp_enabled:
        def autocast_ctx():
            return torch.amp.autocast(device_type="cuda")
    else:
        autocast_ctx = nullcontext

    print(f"pos_weight = {float(pos_weight):.3f}")
    print(
        f"Model parameters: {sum(p.numel() for p in physics_model.parameters() if p.requires_grad)/1e6:.2f} M"
    )
    print(f"Loss config: {bundle['loss_config']}")

        
    # =========================================================
    # Train / Eval loops for PhysicsPrior UNet
    # =========================================================
    amp_stream = autocast_ctx

    train_loss_hist, val_ap_hist, val_f1_hist, val_thr_hist, val_iou_hist = [], [], [], [], []
    best_val_ap = -1.0
    best_state = None
    best_thr_val = 0.5

    peak_gpu_gb = None
    epoch_times = []
    epoch_tiles = []
    compute_metrics = {
        "param_count": int(sum(p.numel() for p in physics_model.parameters() if p.requires_grad)),
    }

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)


    def _forward_batch(batch):
        X_raw = batch["X_raw"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        feats = feature_builder(X_raw)
        return feats, y


    def train_physics_epoch():
        physics_model.train()
        losses = []
        tiles_seen = 0
        for batch in tqdm(train_loader, desc="train Physics", leave=False):
            feats, y = _forward_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with amp_stream():
                logits = physics_model(feats)
                loss = criterion(logits, y)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            ema_tracker.update(physics_model)
            polyak_tracker.update(physics_model)
            losses.append(loss.item())
            tiles_seen += feats.size(0)
        return float(np.mean(losses)), tiles_seen


    @torch.no_grad()
    def eval_physics(model_obj, loader, desc="eval Physics"):
        model_obj.eval()
        all_p, all_t = [], []
        for batch in tqdm(loader, desc=desc, leave=False):
            feats, y = _forward_batch(batch)
            logits = model_obj(feats)
            p = torch.sigmoid(logits).flatten().cpu().numpy()
            t = y.flatten().cpu().numpy()
            all_p.append(p)
            all_t.append(t)
        p = np.concatenate(all_p)
        t = np.concatenate(all_t)
        if t.sum() == 0:
            return 0.0, 0.0, 0.5, 0.0
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


    for epoch in range(EPOCHS_PHYSICS):
        if use_cuda:
            torch.cuda.synchronize(device)
        elif use_mps:
            torch.mps.synchronize()
        epoch_start = time.perf_counter()
        tr_loss, tiles_seen = train_physics_epoch()
        if use_cuda:
            torch.cuda.synchronize(device)
        elif use_mps:
            torch.mps.synchronize()
        epoch_duration = time.perf_counter() - epoch_start
        epoch_times.append(epoch_duration)
        epoch_tiles.append(tiles_seen)
        ap, f1, thr, iou = eval_physics(physics_model, val_loader)
        train_loss_hist.append(tr_loss)
        val_ap_hist.append(ap)
        val_f1_hist.append(f1)
        val_thr_hist.append(thr)
        val_iou_hist.append(iou)
        print(
            f"[Physics] Epoch {epoch:02d} | loss {tr_loss:.4f} | VAL AP {ap:.4f} | VAL F1* {f1:.4f} | VAL IoU {iou:.4f} | thr≈{thr:.3f}"
        )
        if ap > best_val_ap:
            best_val_ap = ap
            best_state = {k: v.cpu().clone() for k, v in physics_model.state_dict().items()}

    if best_state is not None:
        physics_model.load_state_dict(best_state)

    if val_thr_hist:
        best_thr_val = val_thr_hist[-1]

    if epoch_times:
        avg_epoch = float(np.mean(epoch_times))
        std_epoch = float(np.std(epoch_times))
        total_time = float(np.sum(epoch_times))
        total_tiles = int(np.sum(epoch_tiles)) if epoch_tiles else 0
        throughput = float(total_tiles / total_time) if total_time > 0 else None
    else:
        avg_epoch = std_epoch = throughput = None

    if use_cuda:
        torch.cuda.synchronize(device)
        peak_gpu_gb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))
    elif use_mps:
        torch.mps.synchronize()

    @torch.no_grad()
    def measure_latency(ds, model_obj, repeats=50):
        if len(ds) == 0:
            return None
        model_obj.eval()
        sample = ds[0]["X_raw"].unsqueeze(0).to(device)
        feats = feature_builder(sample)
        if use_cuda:
            torch.cuda.synchronize(device)
        elif use_mps:
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            torch.sigmoid(model_obj(feats))
        if use_cuda:
            torch.cuda.synchronize(device)
        elif use_mps:
            torch.mps.synchronize()
        return (time.perf_counter() - start) / repeats

    latency_s = measure_latency(test_ds, physics_model, repeats=100)

    compute_metrics.update({
        "avg_epoch": avg_epoch,
        "std_epoch": std_epoch,
        "throughput_tiles_per_s": throughput,
        "peak_gpu_gb": peak_gpu_gb,
        "latency_s": latency_s,
    })

    def _format_metric(val, unit=None, precision=3):
        if val is None:
            return "—"
        if isinstance(val, (int, np.integer)) and unit is None:
            return f"{int(val)}"
        if isinstance(val, (float, np.floating)):
            if np.isnan(val):
                return "—"
            if unit == "ms":
                return f"{val * 1e3:.{precision}f} {unit}"
            if unit == "GB":
                return f"{val:.{precision}f} {unit}"
            return f"{val:.{precision}f}{'' if unit is None else ' ' + unit}"
        return str(val)

    compute_metrics_display = {
        "Learnable parameters": _format_metric(compute_metrics.get("param_count")),
        "Avg. epoch wall time": _format_metric(compute_metrics.get("avg_epoch"), unit="s"),
        "Epoch time stdev": _format_metric(compute_metrics.get("std_epoch"), unit="s"),
        "Training throughput": _format_metric(compute_metrics.get("throughput_tiles_per_s"), unit="tiles/s"),
        "Peak GPU memory": _format_metric(compute_metrics.get("peak_gpu_gb"), unit="GB"),
        "Inference latency (1 tile)": _format_metric(compute_metrics.get("latency_s"), unit="ms"),
    }

    print("\n[Physics] Computation metrics summary:")
    for k, v in compute_metrics_display.items():
        print(f"  {k:28s} {v}")

    def _tracker_shadow_state(tracker):
        if tracker is None or not hasattr(tracker, "shadow"):
            return None
        return {k: v.detach().cpu() for k, v in tracker.shadow.items()}

    ema_state_dict = _tracker_shadow_state(ema_tracker)
    polyak_state_dict = _tracker_shadow_state(polyak_tracker)

    #save training
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "physicsmodel": physics_model.state_dict(),
            "meanC": meanC.cpu(),
            "stdC": stdC.cpu(),
            "ema_state_dict": ema_state_dict,
            "polyak_state_dict": polyak_state_dict,
            "channels": train_ds.channels,
            "train_loss_hist": train_loss_hist,
            "val_ap_hist": val_ap_hist,
            "val_f1_hist": val_f1_hist,
            "val_iou_hist": val_iou_hist,
            "val_thr_hist": val_thr_hist,
            "best_thr": float(best_thr_val),
            "compute_metrics": compute_metrics,
        },
        output_path,
    )
    print(f"Training complete. Artifact saved to {output_path.resolve()}")
if __name__ == "__main__":
    main()