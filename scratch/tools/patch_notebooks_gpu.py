import json
import sys
from pathlib import Path

GPU_LOADER = '''def make_loader(ds, batch_size=16, upweight_positive=False, shuffle=False, num_workers=4):
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
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=True,
        prefetch_factor=2,
    )
'''

TRAIN_LR = '''def train_lr_epoch():
    lr_model.train()
    losses=[]
    for b in tqdm(train_loader, desc="train LR", leave=False):
        X_raw0 = b["X_raw"].to(device, non_blocking=True)
        y      = b["y"].to(device, non_blocking=True)
        X = build_lr_input(X_raw0)
        logits = lr_model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))
'''

EVAL_GENERIC = '''@torch.no_grad()
def eval_generic(model, loader):
    model.eval()
    all_p, all_t = [], []
    for b in tqdm(loader, desc="eval", leave=False):
        X_raw0 = b["X_raw"].to(device, non_blocking=True)
        y      = b["y"].to(device, non_blocking=True)
        X = build_input_for_net(X_raw0) if 'build_input_for_net' in globals() else build_lr_input(X_raw0)
        p = torch.sigmoid(model(X)).flatten().cpu().numpy()
        t = y.flatten().cpu().numpy()
        all_p.append(p); all_t.append(t)
    p = np.concatenate(all_p); t = np.concatenate(all_t)
    return p, t
'''

TRAIN_UNET_AMP = '''scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

def train_one_epoch(model, loader, optim):
    model.train(); prior.eval() if 'prior' in globals() else None
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        X_raw0 = batch["X_raw"].to(device, non_blocking=True)
        y      = batch["y"].to(device, non_blocking=True)
        # optional geom aug
        if 'aug_geo' in globals():
            X_raw0, y = aug_geo(X_raw0, y, train_ds.channels)
        with torch.cuda.amp.autocast(enabled=use_cuda):
            X = build_input_for_net(X_raw0)
            logits = model(X)
            loss = loss_fn(logits, y)
        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        losses.append(loss.item())
    return float(np.mean(losses))
'''

def patch_notebook(path: Path):
    nb = json.loads(path.read_text(encoding='utf-8'))
    changed = False

    # Ensure cuDNN benchmark cell exists
    cudnn_line = "torch.backends.cudnn.benchmark = True"
    has_cudnn = any(
        c.get('cell_type')=='code' and cudnn_line in ''.join(c.get('source',[]))
        for c in nb.get('cells', [])
    )
    if not has_cudnn:
        cell = {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import torch\n",
                "# Enable cuDNN autotuner for optimal conv performance\n",
                "torch.backends.cudnn.benchmark = True\n"
            ],
            "outputs": [],
            "execution_count": None,
        }
        nb['cells'].insert(0, cell)
        changed = True

    for c in nb.get('cells', []):
        if c.get('cell_type') != 'code':
            continue
        src = ''.join(c.get('source', []))
        if 'def make_loader' in src:
            c['source'] = [GPU_LOADER]
            changed = True
        # Patch LogReg train loop
        if 'def train_lr_epoch' in src:
            c['source'] = [TRAIN_LR]
            changed = True
        # Patch UNet AMP train loop
        if 'def train_one_epoch' in src and 'autocast' not in src:
            c['source'] = [TRAIN_UNET_AMP]
            changed = True
        # Make eval non-blocking if present
        if 'def eval_lr' in src:
            c['source'] = [src.replace('.to(device)', '.to(device, non_blocking=True)')]
            changed = True
        if 'def evaluate(' in src and '.to(device, non_blocking=True)' not in src:
            c['source'] = [src.replace('.to(device)', '.to(device, non_blocking=True)')]
            changed = True

    if changed:
        path.write_text(json.dumps(nb, indent=1), encoding='utf-8')
    return changed

if __name__ == '__main__':
    any_changed = False
    for nb_path in sys.argv[1:]:
        p = Path(nb_path)
        ok = patch_notebook(p)
        print(f"Patched {p}: {ok}")
        any_changed = any_changed or ok
    sys.exit(0 if any_changed else 0)

