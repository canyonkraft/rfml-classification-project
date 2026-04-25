"""
Transfer Learning: RML2016.10a → Panoradio HF
----------------------------------------------
Takes the RML-pretrained CNN feature extractor and fine-tunes it on the
Panoradio HF dataset with a new 18-class output head.

This script tests the central transfer learning question:
  "Do the convolutional features learned on synthetic RML2016.10a signals
   provide a useful starting point for classifying real-style HF signals?"

Three experiments are run for comparison:
  1. SCRATCH      — randomly initialized model trained on Panoradio
                    (baseline: how good can we get with no pre-training?)
  2. FROZEN       — RML conv layers frozen, only the new head trained
                    (tests whether RML features are directly useful)
  3. FINE-TUNED   — RML conv layers initialized then trained at low LR,
                    head trained at higher LR
                    (the standard transfer learning recipe)

Outputs:
  - panoradio_finetune_<MODE>.pth         saved checkpoints
  - panoradio_finetune_comparison.png     accuracy curves for all 3 runs
  - panoradio_finetune_confusion_<MODE>.png  confusion matrix per mode
  - Per-class F1 scores printed for each mode

Key handling differences from eval_panoradio.py:
  - Panoradio signals are RESAMPLED (not truncated) to match RML's 128-sample
    vector length, preserving the signal structure across the segment.
  - Each Panoradio segment is independently normalized (zero-mean, unit-var),
    so the model sees consistent input scale regardless of source dataset.

Usage:
  python3 finetune_panoradio.py                        # run all 3 experiments
  python3 finetune_panoradio.py --mode finetune        # just fine-tuning
  python3 finetune_panoradio.py --epochs 15 --subset 30000   # quick test
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
from scipy.signal import resample
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--npy",        default="dataset_panoradio_hf.npy")
parser.add_argument("--csv",        default="dataset_panoradio_hf_tags.csv")
parser.add_argument("--checkpoint", default="radioml_cnn.pth",
                    help="RML pre-trained checkpoint")
parser.add_argument("--mode",       default="all",
                    choices=["all", "scratch", "frozen", "finetune"])
parser.add_argument("--epochs",     type=int, default=20)
parser.add_argument("--batch",      type=int, default=256)
parser.add_argument("--subset",     type=int, default=0,
                    help="Use only N samples (0 = use all). Useful for testing.")
args = parser.parse_args()

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED   = 42
RML_SAMPLE_LEN = 128       # target length to match RML model input
PANORADIO_LEN  = 2048      # native Panoradio sample length

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Device: {DEVICE}  |  Mode: {args.mode}  |  Epochs: {args.epochs}\n")


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class PanoradioDataset(Dataset):
    """
    Panoradio HF dataset, prepared for the RML-shaped CNN.

    Steps applied per sample:
      1. Resample 2048 complex points → 128 complex points (preserves structure
         far better than truncation; uses scipy.signal.resample which performs
         FFT-based polyphase resampling)
      2. Split into (2, 128) real/imag channels
      3. Per-sample standardization (zero-mean, unit-variance per channel)

    Per-sample normalization is appropriate here because Panoradio signals
    are already power-normalized to 1, so no global stats are needed.
    """

    def __init__(self, npy_path: str, csv_path: str, subset: int = 0):
        print(f"Loading Panoradio HF (memory-mapped) from {npy_path} ...")
        raw = np.load(npy_path, mmap_mode="r")
        print(f"  Raw shape: {raw.shape}  dtype: {raw.dtype}")

        tags = pd.read_csv(csv_path)
        tags.columns = [c.strip().lower() for c in tags.columns]

        N      = raw.shape[0]
        modes  = tags["mode"].values
        snrs   = tags["snr"].values.astype(np.float32)

        unique_modes = sorted(set(modes))
        self.label_map     = {m: i for i, m in enumerate(unique_modes)}
        self.class_names   = unique_modes
        self.num_classes   = len(unique_modes)
        print(f"  {self.num_classes} classes: {unique_modes}")

        # Optional subsetting for fast iteration / debugging
        if subset > 0 and subset < N:
            rng = np.random.default_rng(RANDOM_SEED)
            sel = rng.choice(N, subset, replace=False)
            sel.sort()                 # mmap likes ascending access
            print(f"  Subsetting to {subset:,} random samples")
        else:
            sel = np.arange(N)

        # Pre-resample everything in chunks to avoid loading 5GB at once.
        # Output: (selected_N, 2, 128) float32
        chunk = 2048
        out   = np.zeros((len(sel), 2, RML_SAMPLE_LEN), dtype=np.float32)
        print(f"  Resampling 2048 → {RML_SAMPLE_LEN} samples per signal "
              f"(chunked, ~{len(sel)//chunk + 1} chunks) ...")
        for start in range(0, len(sel), chunk):
            end       = min(start + chunk, len(sel))
            batch_idx = sel[start:end]
            batch     = np.array(raw[batch_idx])     # forces mmap → mem
            # FFT-resample each row to 128 complex samples
            resampled = resample(batch, RML_SAMPLE_LEN, axis=1)
            out[start:end, 0] = resampled.real.astype(np.float32)
            out[start:end, 1] = resampled.imag.astype(np.float32)
            if (start // chunk) % 10 == 0:
                pct = 100.0 * end / len(sel)
                print(f"    progress: {pct:5.1f}%", end="\r", flush=True)
        print(f"    progress: 100.0%")

        # Per-sample normalization (per-channel zero-mean unit-variance)
        mean = out.mean(axis=2, keepdims=True)
        std  = out.std (axis=2, keepdims=True)
        std  = np.where(std < 1e-8, 1.0, std)
        out  = (out - mean) / std

        self.X    = out
        self.Y    = np.array([self.label_map[m] for m in modes[sel]],
                              dtype=np.int64)
        self.SNR  = snrs[sel]
        self.MODE = modes[sel]
        print(f"  Final shape: {self.X.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx]),
                torch.tensor(self.Y[idx]),
                torch.tensor(self.SNR[idx]))


def split_indices(n, train=0.6, val=0.2, seed=42):
    rng     = np.random.default_rng(seed)
    idx     = rng.permutation(n)
    n_train = int(n * train)
    n_val   = int(n * val)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


# ─────────────────────────────────────────────
# Model — same architecture as RML, swappable head
# ─────────────────────────────────────────────
class RadioMLCNN(nn.Module):
    """
    Same architecture as train_radioml.py, but the classifier head's final
    Linear layer is sized to `num_classes` so it can accept new class counts
    when loading RML weights (the head is replaced).
    """

    def __init__(self, num_classes: int):
        super().__init__()

        def conv_bn_relu(in_ch, out_ch, k=3):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_bn_relu(2,   64),  conv_bn_relu(64,  64),  nn.MaxPool1d(2),
            conv_bn_relu(64,  128), conv_bn_relu(128, 128), nn.MaxPool1d(2),
            conv_bn_relu(128, 128), conv_bn_relu(128, 128), nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128),      nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(mode: str, num_classes: int, rml_checkpoint: str):
    """
    Build a model in one of three configurations:
      - scratch:  fresh weights, train everything
      - frozen:   load RML weights, FREEZE conv layers, train only new head
      - finetune: load RML weights, train everything (head at higher LR)

    Returns the model and a list of parameter groups for the optimizer.
    """
    model = RadioMLCNN(num_classes).to(DEVICE)

    if mode == "scratch":
        print("  Mode: SCRATCH  (random init, full training)")
        param_groups = [{"params": model.parameters(), "lr": 5e-4}]
        return model, param_groups

    # Load RML weights, but skip the final layer (different num_classes)
    print(f"  Loading RML weights from {rml_checkpoint}")
    rml_state = torch.load(rml_checkpoint, map_location=DEVICE)

    # The final Linear layer has shape (11, 128) in RML, (18, 128) here.
    # Drop it so PyTorch doesn't complain about size mismatch.
    final_key_w = "classifier.7.weight"
    final_key_b = "classifier.7.bias"
    rml_state.pop(final_key_w, None)
    rml_state.pop(final_key_b, None)

    missing, unexpected = model.load_state_dict(rml_state, strict=False)
    print(f"    Loaded successfully")
    print(f"    Replaced final layer: {final_key_w} → 11→{num_classes} classes")
    if unexpected:
        print(f"    Ignored unexpected keys: {unexpected}")

    if mode == "frozen":
        print("  Mode: FROZEN  (conv layers frozen, only head trains)")
        for p in model.features.parameters():
            p.requires_grad = False
        param_groups = [{"params": model.classifier.parameters(), "lr": 5e-4}]

    elif mode == "finetune":
        print("  Mode: FINE-TUNE  (conv layers trained at low LR, head at high LR)")
        param_groups = [
            {"params": model.features.parameters(),   "lr": 5e-5},  # 10x lower
            {"params": model.classifier.parameters(), "lr": 5e-4},
        ]

    return model, param_groups


# ─────────────────────────────────────────────
# Training / eval helpers
# ─────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y, _ in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X, y, _ in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def gather_predictions(model, loader):
    model.eval()
    preds, labels, snrs = [], [], []
    for X, y, snr in loader:
        logits = model(X.to(DEVICE))
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
        snrs.extend(snr.numpy())
    return np.array(preds), np.array(labels), np.array(snrs)


# ─────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────
def plot_comparison(histories):
    """histories = {mode_name: {"train_acc": [...], "val_acc": [...]}}"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"scratch": "tomato", "frozen": "goldenrod", "finetune": "steelblue"}
    for mode, h in histories.items():
        epochs = range(1, len(h["train_acc"]) + 1)
        ax1.plot(epochs, h["train_acc"], "--", color=colors.get(mode, "black"),
                 alpha=0.5, label=f"{mode} (train)")
        ax1.plot(epochs, h["val_acc"],   "-",  color=colors.get(mode, "black"),
                 linewidth=2, label=f"{mode} (val)")
        ax2.plot(epochs, h["val_loss"],  "-",  color=colors.get(mode, "black"),
                 linewidth=2, label=f"{mode}")

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Validation Accuracy by Training Mode"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss by Training Mode"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("panoradio_finetune_comparison.png", dpi=150)
    print("Saved panoradio_finetune_comparison.png")


def plot_confusion(preds, labels, class_names, title, fname):
    cm      = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")


def plot_snr_curves(snr_results):
    """snr_results = {mode: {snr: accuracy}}"""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"scratch": "tomato", "frozen": "goldenrod", "finetune": "steelblue"}
    for mode, snr_acc in snr_results.items():
        snrs = sorted(snr_acc.keys())
        accs = [snr_acc[s] for s in snrs]
        ax.plot(snrs, accs, marker="o", linewidth=2,
                color=colors.get(mode, "black"), label=mode)
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Accuracy")
    ax.set_title("Test Accuracy vs SNR — Transfer Learning Comparison")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig("panoradio_finetune_snr_curves.png", dpi=150)
    print("Saved panoradio_finetune_snr_curves.png")


# ─────────────────────────────────────────────
# Run a single training experiment
# ─────────────────────────────────────────────
def run_experiment(mode: str, dataset, train_loader, val_loader, test_loader):
    print(f"\n{'═' * 70}")
    print(f"  EXPERIMENT: {mode.upper()}")
    print(f"{'═' * 70}")

    model, param_groups = build_model(mode, dataset.num_classes, args.checkpoint)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {n_train:,} / {n_total:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = 0.0
    save_path = f"panoradio_finetune_{mode}.pth"
    history   = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    print(f"\n  Training for {args.epochs} epochs ...\n")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate(model,   val_loader,   criterion)
        scheduler.step()

        history["train_acc"].append(tr_acc);  history["val_acc"].append(vl_acc)
        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)

        marker = " ← best" if vl_acc > best_val else ""
        if vl_acc > best_val:
            best_val = vl_acc
            torch.save(model.state_dict(), save_path)

        print(f"  Epoch {epoch:>3}/{args.epochs}  "
              f"train: loss {tr_loss:.4f}  acc {tr_acc:.4f}  |  "
              f"val: loss {vl_loss:.4f}  acc {vl_acc:.4f}{marker}")

    # Test
    print(f"\n  Loading best checkpoint (val acc = {best_val:.4f}) ...")
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    te_loss, te_acc = evaluate(model, test_loader, criterion)
    print(f"  Test loss: {te_loss:.4f}  |  Test accuracy: {te_acc:.4f}")

    # Detailed analysis
    preds, labels, snrs = gather_predictions(model, test_loader)
    plot_confusion(preds, labels, dataset.class_names,
                   f"Confusion Matrix — {mode.upper()} (test acc = {te_acc:.3f})",
                   f"panoradio_finetune_confusion_{mode}.png")

    print(f"\n  Per-class F1 ({mode}):")
    print(classification_report(labels, preds,
                                target_names=dataset.class_names,
                                zero_division=0, digits=3))

    # Per-SNR accuracy
    snr_acc = {}
    for snr in sorted(set(snrs)):
        mask          = snrs == snr
        snr_acc[snr]  = float((preds[mask] == labels[mask]).mean())
    print(f"  Per-SNR accuracy ({mode}):")
    for snr, acc in sorted(snr_acc.items()):
        print(f"    {snr:>+5.0f} dB  {acc:.3f}")

    return history, snr_acc, te_acc


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    for path, name in [(args.npy, "Panoradio .npy"),
                       (args.csv, "Panoradio CSV")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")
    if args.mode in ("frozen", "finetune", "all") and \
       not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"RML checkpoint not found: {args.checkpoint}\n"
            "Run train_radioml.py first to generate it.")

    # Load and prepare dataset
    dataset = PanoradioDataset(args.npy, args.csv, subset=args.subset)
    train_idx, val_idx, test_idx = split_indices(len(dataset), seed=RANDOM_SEED)
    print(f"\nSplit → train: {len(train_idx):,}  val: {len(val_idx):,}  "
          f"test: {len(test_idx):,}\n")

    def make_loader(idx, shuffle):
        return DataLoader(Subset(dataset, idx), batch_size=args.batch,
                          shuffle=shuffle, num_workers=4, pin_memory=True)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx,   shuffle=False)
    test_loader  = make_loader(test_idx,  shuffle=False)

    # Decide which experiments to run
    modes_to_run = ["scratch", "frozen", "finetune"] if args.mode == "all" \
                   else [args.mode]

    histories   = {}
    snr_results = {}
    test_accs   = {}

    for mode in modes_to_run:
        h, snr_acc, te_acc = run_experiment(mode, dataset,
                                            train_loader, val_loader, test_loader)
        histories[mode]   = h
        snr_results[mode] = snr_acc
        test_accs[mode]   = te_acc

    # Final comparison plots / summary
    if len(modes_to_run) > 1:
        plot_comparison(histories)
        plot_snr_curves(snr_results)

        print(f"\n{'═' * 70}")
        print("  FINAL COMPARISON")
        print(f"{'═' * 70}")
        for mode, acc in test_accs.items():
            print(f"  {mode:<12}  test accuracy: {acc:.4f}")

        # Compare scratch vs finetune to quantify transfer learning benefit
        if "scratch" in test_accs and "finetune" in test_accs:
            delta = test_accs["finetune"] - test_accs["scratch"]
            sign  = "+" if delta >= 0 else ""
            print(f"\n  Transfer learning impact (finetune − scratch): "
                  f"{sign}{delta:.4f}")
            if delta > 0.01:
                print("  → RML pretraining HELPED")
            elif delta < -0.01:
                print("  → RML pretraining HURT")
            else:
                print("  → RML pretraining had negligible effect")

    print("\nDone.")


if __name__ == "__main__":
    main()
