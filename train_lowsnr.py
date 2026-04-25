"""
RadioML RML2016.10a - Low-SNR Training + Full-SNR Evaluation
-------------------------------------------------------------
Stage 1: Train on low-SNR samples only (default: SNR <= 0dB)
Stage 2: Evaluate the trained model across ALL SNR levels to
         visualise how well low-SNR knowledge generalises upward.

Outputs:
  - lowsnr_cnn.pth 
  - lowsnr_training_history.png 
  - lowsnr_snr_accuracy.png 
  - lowsnr_confusion.png

Usage:
  python3 train_lowsnr.py
  python3 train_lowsnr.py --snr_threshold -4    # only train on SNR <= -4 dB
"""

import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--snr_threshold", type=int, default=0,
                    help="Train only on SNR <= this value (dB). Default: 0")
parser.add_argument("--dataset",  default="RML2016.10a_dict.pkl")
parser.add_argument("--save",     default="lowsnr_cnn.pth")
parser.add_argument("--epochs",   type=int, default=30)
parser.add_argument("--batch",    type=int, default=256)
parser.add_argument("--lr",       type=float, default=5e-4)
args = parser.parse_args()

DATASET_PATH  = args.dataset
SAVE_PATH     = args.save
SNR_THRESHOLD = args.snr_threshold
BATCH_SIZE    = args.batch
EPOCHS        = args.epochs
LR            = args.lr
RANDOM_SEED   = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"Training on SNR <= {SNR_THRESHOLD} dB  |  device: {DEVICE}")


#Dataset
class RadioMLDatasetSNR(Dataset):
    """
    Full dataset with SNR labels retained per sample.
    Normalization is fit on the training subset to avoid leakage.
    """

    def __init__(self, pkl_path: str):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        mods = sorted({mod for (mod, _) in data.keys()})
        self.label_map   = {mod: i for i, mod in enumerate(mods)}
        self.class_names = mods
        self.snr_values  = sorted({snr for (_, snr) in data.keys()})

        xs, ys, snrs = [], [], []
        for (mod, snr), samples in data.items():
            n = len(samples)
            xs.append(samples.astype(np.float32))
            ys.extend([self.label_map[mod]] * n)
            snrs.extend([snr] * n)

        self.X   = np.concatenate(xs, axis=0)
        self.Y   = np.array(ys,   dtype=np.int64)
        self.SNR = np.array(snrs, dtype=np.int32)

        # normalisation stats — populated by fit_normalize()
        self.mean = np.zeros((2, 1), dtype=np.float32)
        self.std  = np.ones ((2, 1), dtype=np.float32)

        print(f"Loaded {len(self.X):,} total samples | {len(mods)} classes")
        print(f"SNR range in dataset: {min(self.snr_values)} to "
              f"{max(self.snr_values)} dB")

    def fit_normalize(self, indices):
        """Compute mean/std from a subset of indices (training set only)."""
        subset    = self.X[indices]
        self.mean = subset.mean(axis=(0, 2), keepdims=True)[0]
        self.std  = subset.std (axis=(0, 2), keepdims=True)[0]
        self.std  = np.where(self.std < 1e-8, 1.0, self.std)
        print(f"Normalization  mean={self.mean.ravel()}  "
              f"std={self.std.ravel()}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mean) / self.std
        return torch.tensor(x), torch.tensor(self.Y[idx]), \
               torch.tensor(self.SNR[idx])


def make_splits(dataset, snr_threshold, train=0.6, val=0.2):
    """
    Split strategy
    ──────────────
    LOW-SNR pool   (SNR <= snr_threshold)
    60% train  |  20% val  |  20% low-SNR test

    The test set used for the confusion matrix is the low-SNR 20%.
    The full-SNR evaluation combines low-SNR test + all high-SNR samples.
    """
    all_idx  = np.arange(len(dataset))
    low_mask = dataset.SNR <= snr_threshold
    hi_mask  = ~low_mask

    low_idx = all_idx[low_mask]
    hi_idx  = all_idx[hi_mask]

    # shuffle low-SNR pool
    rng = np.random.default_rng(RANDOM_SEED)
    low_idx = rng.permutation(low_idx)

    n_low   = len(low_idx)
    n_train = int(n_low * train)
    n_val   = int(n_low * val)

    train_idx   = low_idx[:n_train]
    val_idx     = low_idx[n_train:n_train + n_val]
    test_lo_idx = low_idx[n_train + n_val:]   # low-SNR test set
    test_hi_idx = hi_idx                        # ALL high-SNR samples

    print(f"\nLow-SNR  pool (SNR <= {snr_threshold} dB): {n_low:,} samples")
    print(f"  train:         {len(train_idx):,}")
    print(f"  val:           {len(val_idx):,}")
    print(f"  low-SNR test:  {len(test_lo_idx):,}")
    print(f"High-SNR pool (SNR >  {snr_threshold} dB): {len(hi_idx):,} samples "
          f"(eval only)\n")

    dataset.fit_normalize(train_idx)

    return train_idx, val_idx, test_lo_idx, test_hi_idx


# Model
class RadioMLCNN(nn.Module):
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


# Train and eval helpers
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
def run_inference(model, loader):
    """Returns (preds, labels, snrs) as numpy arrays."""
    model.eval()
    preds, labels, snrs = [], [], []
    for X, y, snr in loader:
        logits = model(X.to(DEVICE))
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
        snrs.extend(snr.numpy())
    return np.array(preds), np.array(labels), np.array(snrs)


# Plots
def plot_history(train_accs, val_accs, train_losses, val_losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_accs) + 1)
    ax1.plot(epochs, train_accs,   label="Train")
    ax1.plot(epochs, val_accs,     label="Val")
    ax1.set_title(f"Accuracy (trained SNR ≤ {SNR_THRESHOLD} dB)")
    ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(epochs, train_losses, label="Train")
    ax2.plot(epochs, val_losses,   label="Val")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout()
    plt.savefig("lowsnr_training_history.png", dpi=150)
    print("Saved lowsnr_training_history.png")


def plot_snr_accuracy(snr_values, snr_accs, class_names, snr_class_accs,
                      snr_threshold):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11))

    colors = ["steelblue" if s <= snr_threshold else "tomato"
              for s in snr_values]
    bars = ax1.bar(snr_values, snr_accs, color=colors, width=1.6, edgecolor="white")
    ax1.axvline(snr_threshold + 1, color="black", linestyle="--", linewidth=1.5,
                label=f"Training threshold ({snr_threshold} dB)")
    ax1.axhline(1 / len(class_names), color="grey", linestyle=":",
                label=f"Random chance ({1/len(class_names):.2f})")
    ax1.set_xlabel("SNR (dB)"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy vs SNR  —  blue = seen during training, "
                  "red = unseen during training")
    ax1.set_xticks(snr_values)
    ax1.set_ylim(0, 1); ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    # annotate bars
    for snr, acc in zip(snr_values, snr_accs):
        ax1.text(snr, acc + 0.02, f"{acc:.2f}", ha="center",
                 fontsize=7, rotation=45)

    mat = np.array([[snr_class_accs[snr][c] for snr in snr_values]
                    for c in range(len(class_names))])
    sns.heatmap(mat, annot=True, fmt=".2f",
                xticklabels=[str(s) for s in snr_values],
                yticklabels=class_names,
                cmap="RdYlGn", vmin=0, vmax=1, ax=ax2)
    ax2.axvline(snr_values.index(snr_threshold) + 1, color="black",
                linewidth=2, linestyle="--")
    ax2.set_xlabel("SNR (dB)"); ax2.set_ylabel("Modulation")
    ax2.set_title("Per-Class Accuracy vs SNR  "
                  "(dashed line = training threshold)")

    plt.tight_layout()
    plt.savefig("lowsnr_snr_accuracy.png", dpi=150)
    print("Saved lowsnr_snr_accuracy.png")


def plot_confusion(preds, labels, class_names, title, fname):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print(f"Saved {fname}")


def print_snr_table(snr_values, snr_accs, class_names, snr_class_accs,
                    snr_threshold):
    sep = "─" * (8 + 10 + 10 * len(class_names))
    print(f"\n{sep}")
    print(f"Per-SNR Accuracy  (training threshold: SNR <= {snr_threshold} dB)")
    print(sep)
    header = f"{'SNR':>6}  {'Overall':>8}  " + \
             "  ".join(f"{c[:7]:>7}" for c in class_names)
    print(header)
    print(sep)
    for snr, acc in zip(snr_values, snr_accs):
        tag = "  [UNSEEN]" if snr > snr_threshold else ""
        row = f"{snr:>+5}dB  {acc:>8.3f}  "
        row += "  ".join(f"{snr_class_accs[snr][c]:>7.3f}"
                         for c in range(len(class_names)))
        print(row + tag)
    print(sep)

def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    dataset = RadioMLDatasetSNR(DATASET_PATH)
    train_idx, val_idx, test_lo_idx, test_hi_idx = make_splits(
        dataset, SNR_THRESHOLD)

    def make_loader(idx, shuffle):
        return DataLoader(Subset(dataset, idx), batch_size=BATCH_SIZE,
                          shuffle=shuffle, num_workers=4, pin_memory=True)

    train_loader   = make_loader(train_idx,   shuffle=True)
    val_loader     = make_loader(val_idx,     shuffle=False)
    test_lo_loader = make_loader(test_lo_idx, shuffle=False)
    # full-SNR eval = low-SNR test + all high-SNR samples
    full_eval_idx  = np.concatenate([test_lo_idx, test_hi_idx])
    full_loader    = make_loader(full_eval_idx, shuffle=False)

    # model & loss
    num_classes = len(dataset.class_names)
    model       = RadioMLCNN(num_classes).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # training loop
    best_val_acc = 0.0
    train_accs, val_accs, train_losses, val_losses = [], [], [], []

    print(f"Training for {EPOCHS} epochs on {DEVICE} …\n")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate(model,   val_loader,   criterion)
        scheduler.step()

        train_accs.append(tr_acc);    val_accs.append(vl_acc)
        train_losses.append(tr_loss); val_losses.append(vl_loss)

        marker = " ← best" if vl_acc > best_val_acc else ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), SAVE_PATH)

        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}  |  "
              f"val loss: {vl_loss:.4f}  acc: {vl_acc:.4f}{marker}")

    # chooses best checkpoint
    print(f"\nLoading best checkpoint (val acc = {best_val_acc:.4f}) …")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))

    # low snr eval
    lo_preds, lo_labels, lo_snrs = run_inference(model, test_lo_loader)
    lo_acc = (lo_preds == lo_labels).mean()
    print(f"\nLow-SNR test accuracy (SNR <= {SNR_THRESHOLD} dB): {lo_acc:.4f}")
    plot_confusion(lo_preds, lo_labels, dataset.class_names,
                   f"Confusion Matrix — Low-SNR Test Set (SNR ≤ {SNR_THRESHOLD} dB)",
                   "lowsnr_confusion.png")

    # eval
    print("\nRunning full-SNR evaluation (all SNR levels) …")
    preds, labels, snrs = run_inference(model, full_loader)

    snr_values     = dataset.snr_values
    snr_accs       = []
    snr_class_accs = {}

    for snr in snr_values:
        mask = snrs == snr
        p, l = preds[mask], labels[mask]
        snr_accs.append(float((p == l).mean()) if mask.sum() > 0 else 0.0)

        per_class = {}
        for c in range(num_classes):
            cm = labels[mask] == c
            if cm.sum() == 0:
                per_class[c] = 0.0
            else:
                per_class[c] = float(((preds[mask] == c) & cm).sum() / cm.sum())
        snr_class_accs[snr] = per_class

    print_snr_table(snr_values, snr_accs, dataset.class_names,
                    snr_class_accs, SNR_THRESHOLD)
    plot_snr_accuracy(snr_values, snr_accs, dataset.class_names,
                      snr_class_accs, SNR_THRESHOLD)

    print("\nOverall Classification Report (all SNRs combined):")
    print(classification_report(labels, preds,
                                target_names=dataset.class_names,
                                zero_division=0))

    plot_history(train_accs, val_accs, train_losses, val_losses)


if __name__ == "__main__":
    main()
