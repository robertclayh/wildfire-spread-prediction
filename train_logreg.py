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
    parser.add_argument("--output", default="logreg_final.pt")
    parser.add_argument("--ckpt", default=None, help="Path to checkpoint file")  # <--- add this

    args = parser.parse_args()

    cfg = load_configuration(args.config)
    EPOCHS_LR = cfg["training"]["epochs"]

    channels = cfg["data"]["channels_used"]
    if channels == "auto":
        channels = None

    #data pipeline
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, meanC, stdC = mndws_models.pipeline_hookup(
    CHANNELS_FOR_MODEL=channels,
    BATCH_SIZE=cfg["data"]["batch_size"],
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
    train_loss_hist = []
    for epoch in range(EPOCHS_LR):
        lr_model.train()
        losses = []
        tiles_seen = 0
        for b in tqdm(train_loader, desc="train LR", leave=False):
            X_raw0, y = b["X_raw"].to(device, non_blocking=True), b["y"].to(device, non_blocking=True)
            X = build_lr_input(X_raw0)
            logits = lr_model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            tiles_seen += X_raw0.size(0)
        train_loss_hist.append(float(np.mean(losses)))
        print(f"Epoch {epoch}: loss={np.mean(losses):.4f}, tiles={tiles_seen}")
        #return float(np.mean(losses)), tiles_seen
    
    #save training
    torch.save(
        {"lrmodel": lr_model.state_dict(),
         "meanC": meanC,
         "stdC": stdC,
         "channels": train_ds.channels},
         args.output,
    )
if __name__ == "__main__":
    main()
