"""
RadioML RML2016.10a - AMC (Automatic Modulation Classification) Training Script
Dataset: https://www.deepsig.ai/datasets (can't download here though)
Download dataset through Kaggle
Architecture: CNN with BatchNorm
Split: 60% train / 20% validation / 20% test
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = "RML2016.10a_dict.pkl" 
BATCH_SIZE   = 256
EPOCHS       = 30
LR           = 5e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED  = 42
SAVE_PATH    = "radioml_cnn.pth"

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class RadioMLDataset(Dataset):
    """
    Loads RML2016.10a from its original pickle format.
    Each value is an ndarray of shape (N, 2, 128) — I/Q samples.
    Normalization is applied after the train split is known
    """

    def __init__(self, pkl_path: str, snr_min: int = -20, snr_max: int = 18):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        mods = sorted({mod for (mod, _) in data.keys()})
        self.label_map   = {mod: i for i, mod in enumerate(mods)}
        self.class_names = mods

        xs, ys = [], []
        for (mod, snr), samples in data.items():
            if snr_min <= snr <= snr_max:
                xs.append(samples.astype(np.float32))
                ys.extend([self.label_map[mod]] * len(samples))

        self.X = np.concatenate(xs, axis=0)
        self.Y = np.array(ys, dtype=np.int64)
        self.mean = np.zeros((2, 1), dtype=np.float32)
        self.std  = np.ones ((2, 1), dtype=np.float32)

        print(f"Loaded {len(self.X):,} samples | {len(mods)} classes | device: {DEVICE}")
        print(f"Classes: {mods}")

    def fit_normalize(self, indices):
        """Compute mean/std from training indices only (no data leakage)."""
        subset = self.X[indices]                      
        # mean/std over samples and time, per I/Q channel
        self.mean = subset.mean(axis=(0, 2), keepdims=True)[0]
        self.std  = subset.std (axis=(0, 2), keepdims=True)[0]
        self.std  = np.where(self.std < 1e-8, 1.0, self.std)
        print(f"Normalization  mean={self.mean.ravel()}  std={self.std.ravel()}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mean) / self.std 
        return torch.tensor(x), torch.tensor(self.Y[idx])


def split_dataset(dataset, train=0.6, val=0.2, test=0.2):
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1"
    n       = len(dataset)
    n_train = int(n * train)
    n_val   = int(n * val)
    n_test  = n - n_train - n_val
    splits  = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    # Fit normalization on training indices only
    dataset.fit_normalize(splits[0].indices)
    return splits


#Model
class RadioMLCNN(nn.Module):
    """
    Input:  (batch, 2, 128)
    Output: (batch, num_classes)

    BatchNorm1d after each Conv stabilises training, speeds up convergence
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
            # Block 1
            conv_bn_relu(2,   64),
            conv_bn_relu(64,  64),
            nn.MaxPool1d(2),

            # Block 2
            conv_bn_relu(64,  128),
            conv_bn_relu(128, 128),
            nn.MaxPool1d(2),

            # Block 3
            conv_bn_relu(128, 128),
            conv_bn_relu(128, 128),
            nn.MaxPool1d(2), 
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# training
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
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
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss   = criterion(logits, y)
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


# plot
def plot_history(train_accs, val_accs, train_losses, val_losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_accs) + 1)

    ax1.plot(epochs, train_accs,   label="Train")
    ax1.plot(epochs, val_accs,     label="Val")
    ax1.set_title("Accuracy");  ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, train_losses, label="Train")
    ax2.plot(epochs, val_losses,   label="Val")
    ax2.set_title("Loss");      ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    print("Saved training_history.png")


@torch.no_grad()
def plot_confusion(model, loader, class_names):
    model.eval()
    all_preds, all_labels = [], []
    for X, y in loader:
        logits = model(X.to(DEVICE))
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.numpy())

    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Normalised Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, zero_division=0))

def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATASET_PATH}'.\n"
            "Download RML2016.10a from https://www.deepsig.ai/datasets "
            "and set DATASET_PATH at the top of this script."
        )

    dataset = RadioMLDataset(DATASET_PATH)
    train_ds, val_ds, test_ds = split_dataset(dataset)
    print(f"Split → train: {len(train_ds):,}  val: {len(val_ds):,}  test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # model and loss
    num_classes = len(dataset.class_names)
    model       = RadioMLCNN(num_classes).to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # training loop
    best_val_acc = 0.0
    train_accs, val_accs, train_losses, val_losses = [], [], [], []

    print(f"\nTraining for {EPOCHS} epochs on {DEVICE} …\n")
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

    # eval
    print(f"\nLoading best checkpoint (val acc = {best_val_acc:.4f}) …")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    te_loss, te_acc = evaluate(model, test_loader, criterion)
    print(f"Test loss: {te_loss:.4f}  |  Test accuracy: {te_acc:.4f}")

    # plot
    plot_history(train_accs, val_accs, train_losses, val_losses)
    plot_confusion(model, test_loader, dataset.class_names)


if __name__ == "__main__":
    main()
