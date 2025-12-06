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

def load_configuration(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def main():
    #configure yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="unet_final.pt")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()

    cfg = load_configuration(args.config)
    EPOCHS_PHYSICS = cfg["training"]["epochs"]

    channels_cfg = cfg["data"]["channels_used"]
    if channels_cfg == "auto":
        channels = mndws_dp.USE_CHANNELS
    else:
        channels = channels_cfg

    #data pipeline
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, meanC, stdC = mndws_models.pipeline_hookup(
    CHANNELS_FOR_MODEL=channels,
    BATCH_SIZE=cfg["data"]["batch_size"],
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
            "focal_alpha": cfg["loss"]["focal_alpha"],
            "focal_gamma": cfg["loss"]["focal_gamma"],
            "focal_weight": cfg["loss"]["focal_weight"],  # 0→pure Tversky, 1→pure focal
            "tversky_alpha": cfg["loss"]["tversky_alpha"],
            "tversky_beta": cfg["loss"]["tversky_beta"],
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

    #build model


    # =========================================================
    # 5) Train loops
    # =========================================================

    for epoch in range(EPOCHS_PHYSICS):
        physics_model.train()
        losses = []
        tiles_seen = 0
        for batch in tqdm(train_loader, desc="train Physics", leave=False):
            feats, y = _forward_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
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
        print(f"Epoch {epoch}: loss={np.mean(losses):.4f}, tiles={tiles_seen}")


    #save training
    torch.save(
        {"physicsmodel": physics_model.state_dict(),
         "meanC": meanC,
         "stdC": stdC,
         "ema_tracker": ema_tracker,
         "polyak_tracker": polyak_tracker,
         "channels": train_ds.channels},
         args.output,
    )
if __name__ == "__main__":
    main()