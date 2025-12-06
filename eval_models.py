"""Unified evaluation entry point for the logistic regression and PhysicsPrior UNet models.

Example
-------
Run evaluation for a saved UNet checkpoint and the logistic regression baseline:

    python eval_models.py --config unet_config.yaml --ckpt outputs/unet_final.pt --model-type unet
    python eval_models.py --config logreg_config.yaml --ckpt outputs/logreg_final.pt --model-type lr
"""

import argparse, yaml, torch, numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve
import time
from pathlib import Path

import matplotlib.pyplot as plt
plt.switch_backend("Agg")

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


def safe_torch_load(path, *, map_location=None):
    """Attempt weights_only=True first to avoid pickle warnings, then fall back."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (TypeError, RuntimeError) as err:
        msg = str(err)
        if "weights_only" not in msg:
            raise
        print(f"weights_only=True unsupported for '{path}'. Falling back to default loader.")
        return torch.load(path, map_location=map_location)


def _metrics_from_probs(p, t, *, force_thr=None):
    if t.sum() == 0:
        thr = float(force_thr if force_thr is not None else 0.5)
        return dict(ap=0.0, f1=0.0, iou=0.0, best_thr=thr, used_thr=thr)

    ap = average_precision_score(t, p)
    prec, rec, thr = precision_recall_curve(t, p)
    f1_curve = (2 * prec * rec) / (prec + rec + 1e-8)
    best_idx = int(f1_curve.argmax())
    best_thr = float(thr[best_idx]) if best_idx < len(thr) else 0.5
    thr_used = float(best_thr if force_thr is None else force_thr)

    yhat = (p >= thr_used).astype(np.float32)
    tp = float((yhat * t).sum())
    fp = float((yhat * (1 - t)).sum())
    fn = float((((1 - yhat) * t)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_used = (2 * precision * recall) / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return dict(ap=float(ap), f1=float(f1_used), iou=float(iou), best_thr=float(best_thr), used_thr=thr_used)


def _format_metric_value(val, unit=None):
    if val is None:
        return "—"
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            val = val.item()
        else:
            val = float(val.mean().item())
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return "—"
        val = float(val.ravel()[0])
    if isinstance(val, (float, np.floating)):
        if np.isnan(val):
            return "—"
        if unit == "ms":
            return f"{val * 1e3:.3f} ms"
        return f"{val:.3f}{'' if unit is None else ' ' + unit}"
    if isinstance(val, (int, np.integer)):
        return f"{val}"
    return str(val)


def _print_compute_summary(summary_dict):
    if not summary_dict:
        return
    print("\nCompute metrics:")
    for label, payload in summary_dict.items():
        if isinstance(payload, tuple):
            value, unit = payload
        else:
            value, unit = payload, None
        print(f"  {label:28s} {_format_metric_value(value, unit)}")


def _print_section(title):
    print("\n============================")
    print(f"   {title}")
    print("============================")


def _print_metrics_table(results):
    if not results:
        return
    print("\nValidation/Test metrics:")
    for name, stats in results.items():
        print(
            f"  {name:6s} | VAL AP {stats['val_ap']:.4f} F1 {stats['val_f1']:.4f} IoU {stats['val_iou']:.4f} thr≈{stats['val_thr']:.3f} | "
            f"TEST AP {stats['test_ap']:.4f} F1 {stats['test_f1']:.4f} IoU {stats['test_iou']:.4f} thr≈{stats['test_thr']:.3f}"
        )


def _prepare_plots_dir(root: str, model_type: str, ckpt_path: str) -> Path:
    plots_root = Path(root)
    plots_root.mkdir(parents=True, exist_ok=True)
    sub_name = f"{model_type}_{Path(ckpt_path).stem}"
    out_dir = plots_root / sub_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _plot_history_curves(train_loss, val_ap, val_f1, val_iou, outfile: Path, title: str):
    train_loss = list(train_loss or [])
    if len(train_loss) == 0:
        return
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    epochs = np.arange(1, len(train_loss) + 1)
    axes[0].plot(epochs, train_loss, marker="o")
    axes[0].set_title(f"{title} Train Loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    series_payload = [
        (val_ap, "VAL AP", "o"),
        (val_f1, "VAL F1*", "s"),
        (val_iou, "VAL IoU", "^"),
    ]
    for seq, label, marker in series_payload:
        if seq:
            seq = list(seq)
            epochs_seq = np.arange(1, len(seq) + 1)
            axes[1].plot(epochs_seq, seq, marker=marker, label=label)
    axes[1].set_title(f"{title} Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, alpha=0.3)
    if axes[1].lines:
        axes[1].legend()
    plt.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_pr_curve(probs: np.ndarray, targets: np.ndarray, title: str, outfile: Path):
    if probs is None or targets is None:
        return
    if probs.size == 0 or targets.size == 0 or np.sum(targets) == 0:
        return
    prec, rec, _ = precision_recall_curve(targets, probs)
    ap = average_precision_score(targets, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title} | AP={ap:.3f}")
    ax.grid(True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _confusion_from_probs(probs: np.ndarray, targets: np.ndarray, thr: float):
    if probs is None or targets is None or thr is None:
        return None, None
    if probs.size == 0 or targets.size == 0:
        return None, None
    yhat = (probs >= float(thr)).astype(np.uint8)
    t = targets.astype(np.uint8)
    tp = int(np.logical_and(yhat == 1, t == 1).sum())
    fp = int(np.logical_and(yhat == 1, t == 0).sum())
    tn = int(np.logical_and(yhat == 0, t == 0).sum())
    fn = int(np.logical_and(yhat == 0, t == 1).sum())
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int32)
    return cm, dict(tp=tp, fp=fp, tn=tn, fn=fn)


def _plot_confusion_matrix(cm: np.ndarray, title: str, outfile: Path):
    if cm is None:
        return
    total = cm.sum()
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Fire", "Fire"])
    ax.set_yticklabels(["No Fire", "Fire"])
    for (i, j), val in np.ndenumerate(cm):
        pct = (val / total * 100.0) if total else 0.0
        ax.text(j, i, f"{val}\n{pct:.1f}%", ha="center", va="center", color="black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _maybe_emit_lr_plots(args, ckpt, val_payload, test_payload, val_metrics, test_metrics):
    plots_dir = getattr(args, "plots_dir", None)
    if not plots_dir:
        return
    out_dir = _prepare_plots_dir(plots_dir, "lr", args.ckpt)
    _plot_history_curves(
        ckpt.get("train_loss_hist"),
        ckpt.get("val_ap_hist"),
        ckpt.get("val_f1_hist"),
        ckpt.get("val_iou_hist"),
        outfile=out_dir / "training_curves.png",
        title="Pixel LogReg",
    )

    val_probs, val_targets = val_payload if val_payload else (None, None)
    test_probs, test_targets = test_payload if test_payload else (None, None)

    _plot_pr_curve(val_probs, val_targets, "LogReg Validation PR", out_dir / "val_pr.png")
    _plot_pr_curve(test_probs, test_targets, "LogReg Test PR", out_dir / "test_pr.png")

    val_cm, _ = _confusion_from_probs(val_probs, val_targets, val_metrics.get("best_thr")) if val_metrics else (None, None)
    test_cm, _ = _confusion_from_probs(test_probs, test_targets, test_metrics.get("used_thr")) if test_metrics else (None, None)
    _plot_confusion_matrix(val_cm, "LogReg Validation Confusion", out_dir / "val_confusion.png")
    _plot_confusion_matrix(test_cm, "LogReg Test Confusion", out_dir / "test_confusion.png")


def _maybe_emit_unet_plots(args, ckpt, variant_payloads):
    plots_dir = getattr(args, "plots_dir", None)
    if not plots_dir:
        return
    out_dir = _prepare_plots_dir(plots_dir, "unet", args.ckpt)
    _plot_history_curves(
        ckpt.get("train_loss_hist"),
        ckpt.get("val_ap_hist"),
        ckpt.get("val_f1_hist"),
        ckpt.get("val_iou_hist"),
        outfile=out_dir / "training_curves.png",
        title="PhysicsPrior UNet",
    )
    for name, payload in (variant_payloads or {}).items():
        variant_dir = out_dir / name.lower()
        variant_dir.mkdir(parents=True, exist_ok=True)
        val_probs, val_targets, val_thr = payload.get("val", (None, None, None))
        test_probs, test_targets, test_thr = payload.get("test", (None, None, None))
        _plot_pr_curve(val_probs, val_targets, f"{name} Validation PR", variant_dir / "val_pr.png")
        _plot_pr_curve(test_probs, test_targets, f"{name} Test PR", variant_dir / "test_pr.png")
        val_cm, _ = _confusion_from_probs(val_probs, val_targets, val_thr)
        test_cm, _ = _confusion_from_probs(test_probs, test_targets, test_thr)
        _plot_confusion_matrix(val_cm, f"{name} Validation Confusion", variant_dir / "val_confusion.png")
        _plot_confusion_matrix(test_cm, f"{name} Test Confusion", variant_dir / "test_confusion.png")



# =========================================================
#  Logistic Regression Evaluation
# =========================================================
@torch.no_grad()
def eval_lr(loader, *, model, build_lr_input, desc=None, force_thr=None, return_probs=False):
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

    metrics = _metrics_from_probs(p, t, force_thr=force_thr)
    if return_probs:
        return metrics, p, t
    return metrics



# =========================================================
#  Physics UNet Evaluation
# =========================================================
@torch.no_grad()
def eval_physics(model_obj, forward_batch_fn, loader, desc="eval Physics", force_thr=None, return_probs=False):
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

    metrics = _metrics_from_probs(p, t, force_thr=force_thr)
    if return_probs:
        return metrics, p, t
    return metrics



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
    ckpt = safe_torch_load(args.ckpt, map_location=device)

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
    val_metrics, val_probs, val_targets = eval_lr(
        val_loader,
        model=lr_model,
        build_lr_input=build_lr_input,
        desc="VAL LR",
        return_probs=True,
    )

    print("Running test metrics…")
    test_metrics, test_probs, test_targets = eval_lr(
        test_loader,
        model=lr_model,
        build_lr_input=build_lr_input,
        desc="TEST LR",
        force_thr=val_metrics["best_thr"],
        return_probs=True,
    )

    compute_summary = {
        "Learnable parameters": (param_count, None),
        "Inference latency (1 tile)": (latency_s, "ms"),
        "Peak GPU memory": (peak_gpu, "GB"),
    }

    results = {
        "LogReg": {
            "val_ap": val_metrics["ap"],
            "val_f1": val_metrics["f1"],
            "val_iou": val_metrics["iou"],
            "val_thr": val_metrics["best_thr"],
            "test_ap": test_metrics["ap"],
            "test_f1": test_metrics["f1"],
            "test_iou": test_metrics["iou"],
            "test_thr": test_metrics["used_thr"],
        }
    }

    _print_section("Pixel LogReg Evaluation")
    _print_compute_summary(compute_summary)
    _print_metrics_table(results)
    _maybe_emit_lr_plots(
        args,
        ckpt,
        (val_probs, val_targets),
        (test_probs, test_targets),
        val_metrics,
        test_metrics,
    )



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

    ckpt = safe_torch_load(args.ckpt, map_location=device)

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

    param_count = int(sum(p.numel() for p in physics_model.parameters() if p.requires_grad))
    compute_metrics = ckpt.get("compute_metrics", {})

    def _spawn_variant():
        return mndws_models.PhysicsUNet(in_ch=in_channels, out_ch=1, base=80).to(device)

    # Build variants: Raw + EMA + Polyak
    variants = {"Raw": physics_model}

    ema_state = ckpt.get("ema_state_dict")
    if ema_state:
        m = _spawn_variant()
        m.load_state_dict(ema_state)
        variants["EMA"] = m
    elif ckpt.get("ema_tracker", None) is not None:
        m = _spawn_variant()
        ckpt["ema_tracker"].copy_to(m)
        variants["EMA"] = m

    polyak_state = ckpt.get("polyak_state_dict")
    if polyak_state:
        m = _spawn_variant()
        m.load_state_dict(polyak_state)
        variants["Polyak"] = m
    elif ckpt.get("polyak_tracker", None) is not None:
        m = _spawn_variant()
        ckpt["polyak_tracker"].copy_to(m)
        variants["Polyak"] = m

    # Evaluate
    results = {}
    plot_payloads = {}
    for name, model_obj in variants.items():
        val_metrics, val_probs, val_targets = eval_physics(
            model_obj,
            _forward_batch_unet,
            val_loader,
            desc=f"VAL {name}",
            return_probs=True,
        )
        test_metrics, test_probs, test_targets = eval_physics(
            model_obj,
            _forward_batch_unet,
            test_loader,
            desc=f"TEST {name}",
            force_thr=val_metrics["best_thr"],
            return_probs=True,
        )

        results[name] = dict(
            val_ap=val_metrics["ap"],
            val_f1=val_metrics["f1"],
            val_iou=val_metrics["iou"],
            val_thr=val_metrics["best_thr"],
            test_ap=test_metrics["ap"],
            test_f1=test_metrics["f1"],
            test_iou=test_metrics["iou"],
            test_thr=test_metrics["used_thr"],
        )
        plot_payloads[name] = {
            "val": (val_probs, val_targets, val_metrics["best_thr"]),
            "test": (test_probs, test_targets, test_metrics["used_thr"]),
        }

    compute_summary = {
        "Learnable parameters": (compute_metrics.get("param_count", param_count), None),
        "Avg. epoch wall time": (compute_metrics.get("avg_epoch"), "s"),
        "Epoch time stdev": (compute_metrics.get("std_epoch"), "s"),
        "Training throughput": (compute_metrics.get("throughput_tiles_per_s"), "tiles/s"),
        "Peak GPU memory": (compute_metrics.get("peak_gpu_gb"), "GB"),
        "Inference latency (1 tile)": (compute_metrics.get("latency_s"), "ms"),
    }

    _print_section("PhysicsPrior UNet Evaluation")
    _print_compute_summary(compute_summary)
    _print_metrics_table(results)

    _maybe_emit_unet_plots(args, ckpt, plot_payloads)

    return results



# =========================================================
#  Main: Multi-Checkpoint, Auto Model-Type Detection
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True, nargs="+")
    parser.add_argument("--model-type", choices=["lr", "unet"], default=None)
    parser.add_argument("--plots-dir", default="outputs/plots", help="Directory to store generated plots (set empty to skip)")
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

        sub = argparse.Namespace(config=args.config, ckpt=ck, plots_dir=args.plots_dir)

        if mt == "lr":
            lr_results(sub)
        else:
            unet_results(sub)

        print("========================================\n")


if __name__ == "__main__":
    main()
