"""Train a Sentinel-1 water segmentation U-Net on Sen1Floods11 (hand-labelled chips).

Runs on Colab/Kaggle (T4) or a local CUDA GPU. Produces:
  <out>/model.pt, <out>/water_unet.onnx, <out>/model_meta.json, <out>/metrics.json

Data: Bonafilia et al., "Sen1Floods11: a georeferenced dataset to train and test deep
learning flood algorithms for Sentinel-1", CVPR Workshops 2020. Public bucket gs://sen1floods11.

Usage:
  python train_sen1floods11.py --data-dir data/sen1floods11 --download --epochs 40 --out runs/unet
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import io
import json
import os
import random
import time
import urllib.request
from pathlib import Path

import numpy as np

BUCKET = "https://storage.googleapis.com/sen1floods11/v1.1"
SPLITS = {
    "train": "splits/flood_handlabeled/flood_train_data.csv",
    "valid": "splits/flood_handlabeled/flood_valid_data.csv",
    "test": "splits/flood_handlabeled/flood_test_data.csv",
    "bolivia": "splits/flood_handlabeled/flood_bolivia_data.csv",
}
S1_DIR = "data/flood_events/HandLabeled/S1Hand"
LABEL_DIR = "data/flood_events/HandLabeled/LabelHand"

# Input scaling shared with backend/pipeline/model.py (also written to model_meta.json).
DB_MIN, DB_MAX = -50.0, 1.0
MEAN = [0.6851, 0.5235]
STD = [0.0820, 0.1102]


def _get(url: str) -> bytes:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(url)


def read_split(data_dir: Path, split: str) -> list[tuple[str, str]]:
    path = data_dir / SPLITS[split]
    rows = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].endswith(".tif"):
                rows.append((row[0].strip(), row[1].strip()))
    return rows


def download(data_dir: Path) -> None:
    for rel in SPLITS.values():
        dst = data_dir / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(_get(f"{BUCKET}/{rel}"))
    jobs = []
    for split in SPLITS:
        for s1, lab in read_split(data_dir, split):
            jobs.append(f"{S1_DIR}/{s1}")
            jobs.append(f"{LABEL_DIR}/{lab}")
    jobs = [j for j in sorted(set(jobs)) if not (data_dir / j).exists()]
    print(f"downloading {len(jobs)} files ...")

    def fetch(rel: str) -> None:
        dst = data_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        tmp.write_bytes(_get(f"{BUCKET}/{rel}"))
        tmp.replace(dst)

    with cf.ThreadPoolExecutor(16) as ex:
        for i, _ in enumerate(ex.map(fetch, jobs)):
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}")


def scale_db(x: np.ndarray) -> np.ndarray:
    """dB (2,H,W) -> normalised float32, NaN -> 0 after scaling."""
    x = np.nan_to_num(x, nan=DB_MIN)
    x = (np.clip(x, DB_MIN, DB_MAX) - DB_MIN) / (DB_MAX - DB_MIN)
    for c in range(2):
        x[c] = (x[c] - MEAN[c]) / STD[c]
    return x.astype(np.float32)


def load_pair(data_dir: Path, s1: str, lab: str) -> tuple[np.ndarray, np.ndarray]:
    import rasterio

    with rasterio.open(data_dir / S1_DIR / s1) as src:
        x = src.read().astype(np.float32)
    with rasterio.open(data_dir / LABEL_DIR / lab) as src:
        y = src.read(1).astype(np.int64)
    y[(y != 0) & (y != 1)] = -1
    return scale_db(x), y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/sen1floods11")
    ap.add_argument("--out", default="runs/unet")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true", help="fp32 only (GTX 16xx cards produce NaN in fp16)")
    ap.add_argument("--repeats", type=int, default=4, help="random crops per chip per epoch")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import segmentation_models_pytorch as smp
    from torch.utils.data import DataLoader, Dataset

    data_dir = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.download:
        download(data_dir)

    splits = {k: read_split(data_dir, k) for k in SPLITS}
    cache = {k: [load_pair(data_dir, *p) for p in v] for k, v in splits.items()}
    print({k: len(v) for k, v in cache.items()})

    class Chips(Dataset):
        def __init__(self, items, train):
            self.items, self.train = items, train

        def __len__(self):
            return len(self.items) * (args.repeats if self.train else 1)

        def __getitem__(self, i):
            x, y = self.items[i % len(self.items)]
            if self.train:
                c = args.crop
                r0 = random.randint(0, x.shape[1] - c)
                c0 = random.randint(0, x.shape[2] - c)
                x, y = x[:, r0:r0 + c, c0:c0 + c], y[r0:r0 + c, c0:c0 + c]
                if random.random() < 0.5:
                    x, y = x[:, :, ::-1], y[:, ::-1]
                if random.random() < 0.5:
                    x, y = x[:, ::-1, :], y[::-1, :]
                k = random.randint(0, 3)
                x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
                # radiometric jitter: SAR calibration differs between processors (GEE vs Sentinel Hub)
                x = x + np.random.normal(0, 0.15, size=(2, 1, 1)).astype(np.float32)
            return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    amp = dev == "cuda" and not args.no_amp
    model = smp.Unet(args.encoder, encoder_weights="imagenet", in_channels=2, classes=2).to(dev)
    dice = smp.losses.DiceLoss("multiclass", ignore_index=-1)
    weight = torch.tensor([1.0, 3.0], device=dev)  # water is the minority class
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_dl = DataLoader(Chips(cache["train"], True), batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train_dl))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    @torch.no_grad()
    def evaluate(items) -> dict:
        model.eval()
        tp = fp = fn = tn = 0
        for x, y in items:
            xt = torch.from_numpy(x)[None].to(dev)
            pred = model(xt).argmax(1)[0].cpu().numpy()
            valid = y != -1
            p, t = pred[valid] == 1, y[valid] == 1
            tp += int((p & t).sum()); fp += int((p & ~t).sum())
            fn += int((~p & t).sum()); tn += int((~p & ~t).sum())
        iou = tp / max(tp + fp + fn, 1)
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        return {"iou_water": round(iou, 4), "f1_water": round(f1, 4), "precision": round(prec, 4),
                "recall": round(rec, 4), "accuracy": round(acc, 4), "chips": len(items)}

    best, history = -1.0, []
    for epoch in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(device_type="cuda", enabled=amp):
                logits = model(x)
            # per-pixel CE averaged over labelled pixels only (some crops are entirely no-data)
            n_valid = (y != -1).sum()
            if n_valid == 0:
                continue
            ce = F.cross_entropy(logits.float(), y, weight=weight, ignore_index=-1, reduction="sum") / n_valid
            loss = ce + dice(logits.float(), y)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        val = evaluate(cache["valid"])
        history.append({"epoch": epoch + 1, "loss": tot / len(train_dl), **val})
        print(f"epoch {epoch + 1}/{args.epochs} loss={tot / len(train_dl):.4f} "
              f"val_iou={val['iou_water']:.4f} ({time.time() - t0:.0f}s)")
        if val["iou_water"] > best:
            best = val["iou_water"]
            torch.save(model.state_dict(), out / "model.pt")

    model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    metrics = {split: evaluate(cache[split]) for split in ("valid", "test", "bolivia")}
    metrics["history"] = history
    metrics["encoder"] = args.encoder
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))

    model.eval().cpu()
    dummy = torch.zeros(1, 2, 256, 256)
    torch.onnx.export(model, dummy, out / "water_unet.onnx", input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "b", 2: "h", 3: "w"}, "logits": {0: "b", 2: "h", 3: "w"}},
                      opset_version=17, dynamo=False)
    meta = {"inputs": ["VV_dB", "VH_dB"], "backscatter": "sigma0", "db_min": DB_MIN, "db_max": DB_MAX,
            "mean": MEAN, "std": STD, "classes": ["not_water", "water"], "encoder": args.encoder,
            "train_data": "Sen1Floods11 hand-labelled (Bonafilia et al., CVPR-W 2020)",
            "metrics": {k: metrics[k] for k in ("valid", "test", "bolivia")}}
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
