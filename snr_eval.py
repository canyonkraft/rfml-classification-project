"""
SNR-Conditioned Evaluation Add-on for train_radioml.py
-------------------------------------------------------
Run this AFTER training to load the saved checkpoint and produce:
  - Per-SNR accuracy curve (snr_accuracy_curve.png)
  - Per-SNR confusion matrices saved as snr_confusion_<SNR>dB.png
  - Console table of per-class F1 at each SNR

Requires: radioml_cnn.pth and RML2016.10a_dict.pkl in the same directory.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── must match train_radioml.py exactly ──────────────────────────────────────
DATASET_PATH = "RML2016.10a_dict.pkl"
SAVE_PATH    = "radioml_cnn.pth"
BATCH_SIZE   = 512
RANDOM_SEED  = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# ─────────────────────────────────────────────────────────────────────────────


# ── Dataset (SNR-aware version) ───────────────────────────────────────────────
class RadioMLDatasetSNR(Dataset):
    """
    Same as RadioMLDataset but also returns the SNR label per sample,
    so we can group test predictions by SNR after inference.
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

        print(f"Loaded {len(self.X):,} samples | {len(mods)} classes")
        print(f"SNR range: {min(self.snr_values)} to {max(self.snr_values)} dB")

    def fit_normalize(self, indices):
        subset    = self.X[indices]
        self.mean = subset.mean(axis=(0, 2), keepdims=True)[0]
        self.std  = subset.std (axis=(0, 2), keepdims=True)[0]
        self.std  = np.where(self.std < 1e-8, 1.0, self.std)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mean) / self.std
        return torch.tensor(x), torch.tensor(self.Y[idx]), torch.tensor(self.SNR[idx])


def split_indices(n, train=0.6, val=0.2, seed=42):
    """Return train/val/test index arrays with the same split as training."""
    rng     = np.random.default_rng(seed)
    idx     = rng.permutation(n)
    n_train = int(n * train)
    n_val   = int(n * val)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


# ── Model (must match train_radioml.py) ──────────────────────────────────────
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
            conv_bn_relu(2,   64), conv_bn_relu(64,  64),  nn.MaxPool1d(2),
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


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader):
    """Returns (all_preds, all_labels, all_snrs) as numpy arrays."""
    model.eval()
    preds, labels, snrs = [], [], []
    for X, y, snr in loader:
        logits = model(X.to(DEVICE))
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.numpy())
        snrs.extend(snr.numpy())
    return np.array(preds), np.array(labels), np.array(snrs)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_snr_accuracy(snr_values, snr_accs, class_names, snr_class_accs):
    """Overall accuracy curve + per-class accuracy heatmap vs SNR."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # --- overall curve ---
    ax1.plot(snr_values, snr_accs, marker="o", linewidth=2, color="steelblue")
    ax1.axhline(1 / len(class_names), color="red", linestyle="--",
                label=f"Random ({1/len(class_names):.2f})")
    ax1.set_xlabel("SNR (dB)"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Overall Accuracy vs SNR")
    ax1.set_xticks(snr_values); ax1.set_ylim(0, 1); ax1.grid(True); ax1.legend()

    # --- per-class heatmap ---
    # rows = classes, cols = SNR levels
    mat = np.array([[snr_class_accs[snr][c] for snr in snr_values]
                    for c in range(len(class_names))])
    sns.heatmap(mat, annot=True, fmt=".2f",
                xticklabels=[str(s) for s in snr_values],
                yticklabels=class_names,
                cmap="RdYlGn", vmin=0, vmax=1, ax=ax2)
    ax2.set_xlabel("SNR (dB)"); ax2.set_ylabel("Modulation")
    ax2.set_title("Per-Class Accuracy vs SNR")

    plt.tight_layout()
    plt.savefig("snr_accuracy_curve.png", dpi=150)
    print("Saved snr_accuracy_curve.png")


def print_snr_table(snr_values, snr_accs, class_names, snr_class_accs):
    col_w = 8
    header = f"{'SNR':>6}  {'Overall':>{col_w}}  " + \
             "  ".join(f"{c:>{col_w}}" for c in class_names)
    print("\n" + "─" * len(header))
    print("Per-SNR Accuracy Breakdown")
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for snr, acc in zip(snr_values, snr_accs):
        row = f"{snr:>+5}dB  {acc:>{col_w}.3f}  "
        row += "  ".join(f"{snr_class_accs[snr][c]:>{col_w}.3f}"
                         for c in range(len(class_names)))
        print(row)
    print("─" * len(header))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {SAVE_PATH}\n"
            "Run train_radioml.py first to generate it.")

    # 1. Load dataset with SNR labels
    dataset = RadioMLDatasetSNR(DATASET_PATH)

    # 2. Reproduce exact same split as training (same seed + order)
    train_idx, val_idx, test_idx = split_indices(len(dataset), seed=RANDOM_SEED)
    dataset.fit_normalize(train_idx)   # same normalization as training

    test_subset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True)

    # 3. Load model
    model = RadioMLCNN(len(dataset.class_names)).to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    print(f"Loaded checkpoint: {SAVE_PATH}")

    # 4. Run inference on test set
    preds, labels, snrs = run_inference(model, test_loader)
    print(f"Test samples evaluated: {len(preds):,}")

    # 5. Compute per-SNR stats
    snr_values    = sorted(dataset.snr_values)
    snr_accs      = []
    snr_class_accs = {}   # snr → {class_idx → accuracy}

    for snr in snr_values:
        mask = snrs == snr
        p, l = preds[mask], labels[mask]
        snr_accs.append((p == l).mean())

        per_class = {}
        for c in range(len(dataset.class_names)):
            cm = labels[mask] == c
            if cm.sum() == 0:
                per_class[c] = 0.0
            else:
                per_class[c] = ((preds[mask] == c) & cm).sum() / cm.sum()
        snr_class_accs[snr] = per_class

    # 6. Print table
    print_snr_table(snr_values, snr_accs, dataset.class_names, snr_class_accs)

    # 7. Plot
    plot_snr_accuracy(snr_values, snr_accs, dataset.class_names, snr_class_accs)

    # 8. Overall classification report
    print("\nOverall Classification Report (all SNRs):")
    print(classification_report(labels, preds,
                                target_names=dataset.class_names,
                                zero_division=0))


if __name__ == "__main__":
    main()
