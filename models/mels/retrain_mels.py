import os
import sys
import time
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def main():
    start_time = time.time()
    base_dir = Path(__file__).resolve().parents[2]
    mels_dir = base_dir / "models" / "mels"
    backup_dir = mels_dir / "checkpoints_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STARTING RETRAINING & FINE-TUNING OF CNN SPECTROGRAM MODELS")
    print("=" * 70, flush=True)

    # 1. Ensure original checkpoints are backed up
    for fname in ["resnet18.pth", "efficientnetb0.pth", "mobilenetv2.pth"]:
        src = mels_dir / fname
        dst = backup_dir / fname
        if src.exists() and not dst.exists():
            print(f"Backing up original checkpoint: {fname} -> checkpoints_backup/", flush=True)
            shutil.copy2(src, dst)

    # 2. Load CSV Splits
    data_dir = base_dir / "csvs" / "CSVs"
    print(f"\nLoading CSV datasets from: {data_dir}", flush=True)
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "valid.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    classes = sorted(train_df["Genre"].unique())
    genre_to_idx = {g: i for i, g in enumerate(classes)}
    num_classes = len(classes)

    print(f"Dataset split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Classes ({num_classes}): {classes}", flush=True)

    mel_base = base_dir / "mel_arrays"
    def get_paths_and_labels(df):
        paths = []
        labels = []
        for fn, g in zip(df["Filename"], df["Genre"]):
            base = fn.replace(".wav", "")
            paths.append(str(mel_base / g / f"{base}.npy"))
            labels.append(genre_to_idx[g])
        return paths, np.array(labels)

    train_paths, train_labels = get_paths_and_labels(train_df)
    val_paths, val_labels = get_paths_and_labels(val_df)
    test_paths, test_labels = get_paths_and_labels(test_df)
    test_tracks = test_df["Filename"].apply(
        lambda x: str(x).split("_seg")[0] if "_seg" in str(x) else str(x)
    ).values

    # 3. Setup PyTorch, GPU & Datasets
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import models, transforms
    from torch.utils.data import Dataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    class MelDataset(Dataset):
        def __init__(self, paths, labels, transform=None):
            self.paths = paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            spec = np.load(self.paths[idx])
            if spec.min() >= 0:
                spec = np.log(spec + 1e-9)
            spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=-100.0)
            delta = spec.max() - spec.min()
            spec = (spec - spec.min()) / delta if delta > 1e-6 else np.zeros_like(spec)
            t = torch.from_numpy(spec).float().unsqueeze(0).repeat(3, 1, 1)
            if self.transform:
                t = self.transform(t)
            return t, self.labels[idx]

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    BATCH_SIZE = 32
    train_loader = DataLoader(MelDataset(train_paths, train_labels, train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(MelDataset(val_paths, val_labels, val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(MelDataset(test_paths, test_labels, val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Mixup function
    def mixup_data(x, y, alpha=0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)

    def eval_loader(model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / total

    def predict_probs(model, loader):
        model.eval()
        probs_list = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                out = model(xb)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                probs_list.append(probs)
        return np.vstack(probs_list)

    def train_model(model, name, checkpoint_path, param_groups, epochs=12, patience=3):
        print(f"\nTraining {name} (Max Epochs: {epochs}, Early Stopping: {patience})...", flush=True)
        optimizer = optim.AdamW(param_groups, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_val_acc = eval_loader(model, val_loader)
        print(f"  -> Initial Val Accuracy: {best_val_acc * 100:.2f}%", flush=True)
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0

        for ep in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            t_ep = time.time()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                mixed_x, y_a, y_b, lam = mixup_data(xb, yb, alpha=0.2)
                optimizer.zero_grad()
                out = model(mixed_x)
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            scheduler.step()
            val_acc = eval_loader(model, val_loader)
            train_loss = total_loss / len(train_paths)
            print(f"  Epoch {ep:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | Elapsed: {time.time()-t_ep:.1f}s", flush=True)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
                print(f"    * New best validation accuracy: {best_val_acc * 100:.2f}% (Saved checkpoint)", flush=True)
                torch.save(best_weights, checkpoint_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"    * Early stopping triggered after {ep} epochs.", flush=True)
                    break

        model.load_state_dict(best_weights)
        test_probs = predict_probs(model, test_loader)
        test_preds = test_probs.argmax(axis=1)
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average="weighted")

        # Track Voting
        df_track = pd.DataFrame(test_probs)
        df_track["track"] = test_tracks
        df_track["true"] = test_labels
        track_probs = df_track.groupby("track")[list(range(num_classes))].mean().values
        track_y = df_track.groupby("track")["true"].first().values
        track_acc = accuracy_score(track_y, track_probs.argmax(axis=1))
        print(f"  [{name} Final] Segment Test Acc: {test_acc * 100:.2f}% (F1: {test_f1:.4f}) | Track Acc: {track_acc * 100:.2f}%", flush=True)
        return test_probs

    # --- 1. ResNet-18 ---
    r18 = models.resnet18(weights=None)
    num_ftrs = r18.fc.in_features
    r18.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    r18.load_state_dict(torch.load(mels_dir / "resnet18.pth", map_location="cpu", weights_only=False))
    r18.to(device)

    # Discriminative parameter groups
    backbone_params = [p for n, p in r18.named_parameters() if "fc" not in n and ("layer3" in n or "layer4" in n)]
    head_params = list(r18.fc.parameters())
    for n, p in r18.named_parameters():
        if "layer1" in n or "layer2" in n or "conv1" in n:
            p.requires_grad = False

    r18_groups = [
        {"params": backbone_params, "lr": 2e-5},
        {"params": head_params, "lr": 3e-4}
    ]
    p_r18 = train_model(r18, "ResNet-18", mels_dir / "resnet18.pth", r18_groups, epochs=10, patience=3)
    np.save(mels_dir / "resnet18_test_probs.npy", p_r18)

    # --- 2. EfficientNet-B0 ---
    eff = models.efficientnet_b0(weights=None)
    in_features = eff.classifier[1].in_features
    eff.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    eff.load_state_dict(torch.load(mels_dir / "efficientnetb0.pth", map_location="cpu", weights_only=False))
    eff.to(device)

    eff_backbone = [p for n, p in eff.features.named_parameters() if "6" in n or "7" in n or "8" in n]
    eff_head = list(eff.classifier.parameters())
    for n, p in eff.features.named_parameters():
        if not ("6" in n or "7" in n or "8" in n):
            p.requires_grad = False

    eff_groups = [
        {"params": eff_backbone, "lr": 2e-5},
        {"params": eff_head, "lr": 3e-4}
    ]
    p_eff = train_model(eff, "EfficientNet-B0", mels_dir / "efficientnetb0.pth", eff_groups, epochs=10, patience=3)
    np.save(mels_dir / "efficientnet_test_probs.npy", p_eff)

    # --- 3. MobileNetV2 ---
    mob = models.mobilenet_v2(weights=None)
    in_features = mob.classifier[1].in_features
    mob.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    mob.load_state_dict(torch.load(mels_dir / "mobilenetv2.pth", map_location="cpu", weights_only=False))
    mob.to(device)

    mob_backbone = [p for n, p in mob.features.named_parameters() if any(s in n for s in ["14", "15", "16", "17", "18"])]
    mob_head = list(mob.classifier.parameters())
    for n, p in mob.features.named_parameters():
        if not any(s in n for s in ["14", "15", "16", "17", "18"]):
            p.requires_grad = False

    mob_groups = [
        {"params": mob_backbone, "lr": 2e-5},
        {"params": mob_head, "lr": 3e-4}
    ]
    p_mob = train_model(mob, "MobileNetV2", mels_dir / "mobilenetv2.pth", mob_groups, epochs=10, patience=3)
    np.save(mels_dir / "mobilenetv2_test_probs.npy", p_mob)

    # Save metadata
    np.save(mels_dir / "resnet18_test_labels.npy", test_labels)
    np.save(mels_dir / "resnet18_test_tracks.npy", test_tracks)
    np.save(mels_dir / "resnet18_classes.npy", np.array(classes))

    print("\n" + "=" * 70)
    print("ALL CNN SPECTROGRAM MODELS SUCCESSFULLY RETRAINED & EXPORTED")
    print(f"Total time elapsed: {time.time() - start_time:.1f}s")
    print("=" * 70, flush=True)

if __name__ == '__main__':
    main()
