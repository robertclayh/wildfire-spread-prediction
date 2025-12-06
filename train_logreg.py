"""Train the logistic-regression wildfire baseline.

Launch this script from the repo root to run the shared data pipeline,
fit the pixel-level logistic regression model for a configurable number of
epochs, and save the trained weights plus channel stats to an artifact file.
Use `--config path/to/config.yaml` to pull hyperparameters from YAML or rely
on the CLI flags (epochs, batch size, output path, checkpoint resume).

Example
-------
Train for five epochs with CLI defaults and save to a scratch artifact:

    python train_logreg.py --epochs 5 --output outputs/logreg_smoke_test.pt
"""

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

#our packages
import mNDWS_models as mndws_models
import mNDWS_DataPipeline as mndws_dp

device = mndws_models.device
use_cuda = mndws_models.use_cuda
use_mps = mndws_models.use_mps

def load_configuration(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def main():
    #configure yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to YAML config (optional)")
    parser.add_argument("--output", default="outputs/logreg_final.pt", help="Path to save trained checkpoint")
    parser.add_argument("--ckpt", default=None, help="Path to checkpoint file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")

    args = parser.parse_args()

    # Use config if provided, otherwise use CLI args
    if args.config:
        cfg = load_configuration(args.config)
        EPOCHS_LR = cfg["training"]["epochs"]
        channels = cfg["data"]["channels_used"]
        if channels == "auto":
            channels = mndws_dp.USE_CHANNELS
        batch_size = cfg["data"]["batch_size"]
    else:
        EPOCHS_LR = args.epochs
        channels = mndws_dp.USE_CHANNELS
        batch_size = args.batch_size

    #data pipeline
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, meanC, stdC = mndws_models.pipeline_hookup(
    CHANNELS_FOR_MODEL=channels,
    BATCH_SIZE=batch_size,
 )

    #build model
    lr_model, pw, criterion, optimizer = mndws_models.PixelLogReg_outputs(
        train_ds=train_ds,
        meanC=meanC,
        stdC=stdC,
        train_loader=train_loader,
        device=device,
    )

    if args.ckpt is not None:
            ckpt = torch.load(args.ckpt, map_location=device)
            lr_model.load_state_dict(ckpt["lrmodel"])

    def build_lr_input(X_raw0, mean=None, std=None):
        mean_t = mean if mean is not None else meanC
        std_t = std if std is not None else stdC
        return mndws_models.build_lr_input(X_raw0, mean_t, std_t)
    

    # =========================================================
    # 5) Train loops
    # =========================================================
    train_loss_hist, val_ap_hist, val_f1_hist, val_thr_hist, val_iou_hist = [], [], [], [], []
    best_val_ap = -1.0
    best_state = None
    best_thr_val = 0.5

    peak_gpu_gb = None
    epoch_times = []
    epoch_tiles = []
    compute_metrics = {
        "param_count": int(sum(p.numel() for p in lr_model.parameters() if p.requires_grad)),
    }

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)


    def train_lr_epoch():
        lr_model.train()
        losses = []
        tiles_seen = 0
        for b in tqdm(train_loader, desc="train LR", leave=False):
            X_raw0 = b["X_raw"].to(device, non_blocking=True)
            y = b["y"].to(device, non_blocking=True)
            X = build_lr_input(X_raw0)
            logits = lr_model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            tiles_seen += X_raw0.size(0)
        return float(np.mean(losses)), tiles_seen


    @torch.no_grad()
    def eval_lr(model_obj, loader, desc="eval LR"):
        model_obj.eval()
        all_p, all_t = [], []
        for b in tqdm(loader, desc=desc, leave=False):
            X_raw0 = b["X_raw"].to(device, non_blocking=True)
            y = b["y"].to(device, non_blocking=True)
            logits = model_obj(build_lr_input(X_raw0))
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


    for epoch in range(EPOCHS_LR):
        if use_cuda:
            torch.cuda.synchronize(device)
        elif use_mps:
            torch.mps.synchronize()
        epoch_start = time.perf_counter()
        tr_loss, tiles_seen = train_lr_epoch()
        if use_cuda:
            torch.cuda.synchronize(device)
        elif use_mps:
            torch.mps.synchronize()
        epoch_duration = time.perf_counter() - epoch_start
        epoch_times.append(epoch_duration)
        epoch_tiles.append(tiles_seen)
        ap, f1, thr, iou = eval_lr(lr_model, val_loader)
        train_loss_hist.append(tr_loss)
        val_ap_hist.append(ap)
        val_f1_hist.append(f1)
        val_thr_hist.append(thr)
        val_iou_hist.append(iou)
        print(
            f"[LogReg] Epoch {epoch:02d} | loss {tr_loss:.4f} | VAL AP {ap:.4f} | VAL F1* {f1:.4f} | VAL IoU {iou:.4f} | thr≈{thr:.3f}"
        )
        if ap > best_val_ap:
            best_val_ap = ap
            best_state = {k: v.cpu().clone() for k, v in lr_model.state_dict().items()}

    if best_state is not None:
        lr_model.load_state_dict(best_state)

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
        feats = build_lr_input(sample)
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

    latency_s = measure_latency(test_ds, lr_model, repeats=100)

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

    print("\n[LogReg] Computation metrics summary:")
    for k, v in compute_metrics_display.items():
        print(f"  {k:28s} {v}")

    #save training
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "lrmodel": lr_model.state_dict(),
            "meanC": meanC,
            "stdC": stdC,
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