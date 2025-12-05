#evaluation script that reproduces data comparison tables

#set up
import argparse, yaml, torch, numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve
import time, os


import mNDWS_models as mndws_models

device = mndws_models.device
use_cuda = mndws_models.use_cuda
use_mps = mndws_models.use_mps

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def make_build_lr_input(meanC, stdC):
    def _build(X_raw):
        return mndws_models.build_lr_input(X_raw, meanC, stdC)
    return _build


#logreg
# =========================================================
# Eval loops, change number of epochs above
# =========================================================
@torch.no_grad()
def eval_lr(loader, *, model, build_lr_input, desc=None):
    #model = lr_model if model is None else model
    #model = lrmodel
    model.eval()
    all_p, all_t = [], []

    iter_desc = desc if desc is not None else "eval LR"

    for b in tqdm(loader, desc=iter_desc, leave=False):
        X_raw0, y = b["X_raw"].to(device, non_blocking=True), b["y"].to(device, non_blocking=True)
        X = build_lr_input(X_raw0)
        p = torch.sigmoid(model(X)).flatten().cpu().numpy()
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

def lr_results():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    # ---------------------------------------
    #  Dataset
    # ---------------------------------------
    channels = cfg["data"]["channels_used"]
    if channels == "auto":
        channels = None

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, meanC, stdC = (
        mndws_models.pipeline_hookup(
            CHANNELS_FOR_MODEL=channels,
            BATCH_SIZE=cfg["data"]["batch_size"]
        )
    )

    # ---------------------------------------
    # Load Checkpoint
    # ---------------------------------------
    ckpt = torch.load(args.ckpt, map_location=device)

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


    latency_s = None
    peak_gpu_gb = None

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

        
    @torch.no_grad()
    def measure_latency(ds, repeats=50):
        if len(ds) == 0:
            return None
        lr_model.eval()
        sample = ds[0]["X_raw"].unsqueeze(0).to(device)
        X = build_lr_input(sample)
        if use_cuda:
            torch.cuda.synchronize(device)
        elif use_mps:
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            torch.sigmoid(lr_model(X))
        if use_cuda:
            torch.cuda.synchronize(device)
        elif use_mps:
            torch.mps.synchronize()
        return (time.perf_counter() - start) / repeats

    if latency_s is None:
        latency_s = measure_latency(test_ds, repeats=100)
    if use_cuda and peak_gpu_gb is None:
        peak_gpu_gb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3))


    print("\nRunning validation metrics…")
    val_ap, val_f1, val_thr, val_iou = eval_lr(
        val_loader, model=lr_model, build_lr_input=build_lr_input, desc="VAL LR"
    )

    print("Running test metrics…")
    test_ap, test_f1, test_thr, test_iou = eval_lr(
        test_loader, model=lr_model, build_lr_input=build_lr_input, desc="TEST LR"
    )

    variants = {"Raw": lr_model}

    final_metrics = {}
    for name, model_obj in variants.items():
        ap_val, f1_val, thr_val, iou_val = eval_lr(val_loader, model=model_obj, build_lr_input=build_lr_input, desc=f"VAL {name}")
        ap_test, f1_test, thr_test, iou_test = eval_lr(test_loader, model=model_obj, build_lr_input=build_lr_input, desc=f"TEST {name}")
        final_metrics[name] = {
            "val_ap": ap_val,
            "val_f1": f1_val,
            "val_iou": iou_val,
            "val_thr": thr_val,
            "test_ap": ap_test,
            "test_f1": f1_test,
            "test_iou": iou_test,
            "test_thr": thr_test,
        }

    print("Final metrics (val/test):")
    for name, stats in final_metrics.items():
        print(
            f"  {name:6s} | VAL AP {stats['val_ap']:.4f} F1 {stats['val_f1']:.4f} IoU {stats['val_iou']:.4f} thr≈{stats['val_thr']:.3f}"
            f" | TEST AP {stats['test_ap']:.4f} F1 {stats['test_f1']:.4f} IoU {stats['test_iou']:.4f}",
        )
        
    # Single-feature ablation: evaluate LR using one channel at a time
    single_feats = list(train_ds.channels)  # uses the configured channel order

    @torch.no_grad()
    def eval_single_feature(idx, loader=val_loader):
        C = len(single_feats)
        assert 0 <= idx < C, f"idx out of range (got {idx}, C={C})"

        # Save current weights/bias
        W_orig = lr_model.lin.weight.detach().clone()  # (1,C,1,1)
        b_orig = lr_model.lin.bias.detach().clone()

        # Zero all weights except the selected channel
        W_only = torch.zeros_like(W_orig)
        W_only[0, idx, 0, 0] = W_orig[0, idx, 0, 0]

        lr_model.lin.weight.data.copy_(W_only)
        lr_model.lin.bias.data.copy_(b_orig)  # keep bias unchanged

        # Evaluate on the given loader
        ap, f1, _, _ = eval_lr(loader, model=lr_model, build_lr_input=build_lr_input)

        # Restore original weights/bias
        lr_model.lin.weight.data.copy_(W_orig)
        lr_model.lin.bias.data.copy_(b_orig)

        return ap, f1

    abl = []
    for i, nm in enumerate(single_feats):
        ap_i, f1_i = eval_single_feature(i, loader=val_loader)
        abl.append((nm, ap_i, f1_i))

    abl = sorted(abl, key=lambda x: -x[2])  # sort by F1

    # ================================
    # Output results
    # ================================
    print("\n============================")
    print("   Pixel LogReg Evaluation")
    print("============================")
    print(f"Parameters               : {param_count}")
    print(f"Latency (1 tile)        : {latency_s*1e3:.3f} ms" if latency_s else "—")
    print(f"Peak GPU memory         : {peak_gpu_gb:.3f} GB" if peak_gpu_gb else "—")

    print("\nValidation Metrics:")
    print(f"  AP   : {val_ap:.4f}")
    print(f"  F1   : {val_f1:.4f}")
    print(f"  IoU  : {val_iou:.4f}")
    print(f"  Thr  : {val_thr:.3f}")

    print("\nTest Metrics:")
    print(f"  AP   : {test_ap:.4f}")
    print(f"  F1   : {test_f1:.4f}")
    print(f"  IoU  : {test_iou:.4f}")
    print(f"  Thr  : {test_thr:.3f}")

    print("\nSingle-feature Ablation (sorted by F1):")
    for nm, ap_i, f1_i in abl:
        print(f"  {nm:16s}  AP={ap_i:.3f}  F1={f1_i:.3f}")



def main():
    lr_results()

if __name__ == "__main__":
    main()
