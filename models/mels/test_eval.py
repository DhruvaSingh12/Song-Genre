import os
import sys
import time
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
    print("EVALUATING CNN SPECTROGRAM MODELS ON TEST DATASET")
    print("=" * 70, flush=True)

    # 1. Backup original checkpoints if present
    for fname in ["resnet18.pth", "efficientnetb0.pth", "mobilenetv2.pth"]:
        src = mels_dir / fname
        dst = backup_dir / fname
        if src.exists() and not dst.exists():
            print(f"Backing up original checkpoint: {fname} -> checkpoints_backup/", flush=True)
            import shutil
            shutil.copy2(src, dst)

    # 2. Load test CSV first (before initializing CUDA)
    data_dir = base_dir / "csvs" / "CSVs"
    test_csv = data_dir / "test.csv"
    print(f"\nLoading test CSV from: {test_csv}", flush=True)
    test_df = pd.read_csv(test_csv)
    classes = sorted(test_df["Genre"].unique())
    genre_to_idx = {g: i for i, g in enumerate(classes)}
    y_test = np.array([genre_to_idx[g] for g in test_df["Genre"]])
    test_tracks = test_df["Filename"].apply(
        lambda x: str(x).split("_seg")[0] if "_seg" in str(x) else str(x)
    ).values

    mel_base = base_dir / "mel_arrays"
    file_paths = []
    for fn, g in zip(test_df["Filename"], test_df["Genre"]):
        base = fn.replace(".wav", "")
        file_paths.append(str(mel_base / g / f"{base}.npy"))

    print(f"Test samples: {len(file_paths)} across {len(classes)} classes", flush=True)

    # 3. Initialize PyTorch & GPU
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from torch.utils.data import Dataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    class FastMelDataset(Dataset):
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

    val_transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = FastMelDataset(file_paths, y_test, val_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    def eval_model(model, name):
        model.eval()
        all_probs = []
        t0 = time.time()
        with torch.no_grad():
            for i, (xb, _) in enumerate(test_loader):
                xb = xb.to(device)
                out = model(xb)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                all_probs.append(probs)
                if (i + 1) % 15 == 0 or (i + 1) == len(test_loader):
                    print(f"  [{name}] Evaluated {i+1}/{len(test_loader)} batches ({time.time()-t0:.1f}s)...", flush=True)

        probs = np.vstack(all_probs)
        preds = probs.argmax(axis=1)
        seg_acc = accuracy_score(y_test, preds)
        seg_f1 = f1_score(y_test, preds, average="weighted")

        # Track Voting
        prob_cols = list(range(len(classes)))
        df_track = pd.DataFrame(probs, columns=prob_cols)
        df_track["track"] = test_tracks
        df_track["true"] = y_test
        track_probs = df_track.groupby("track")[prob_cols].mean().values
        track_y_true = df_track.groupby("track")["true"].first().values
        track_preds = track_probs.argmax(axis=1)
        track_acc = accuracy_score(track_y_true, track_preds)
        track_f1 = f1_score(track_y_true, track_preds, average="weighted")

        print(f"  -> {name} Segment Accuracy: {seg_acc * 100:.2f}% (F1: {seg_f1:.4f})", flush=True)
        print(f"  -> {name} Track Accuracy  : {track_acc * 100:.2f}% (F1: {track_f1:.4f})", flush=True)
        return probs, seg_acc, track_acc

    # --- 1. ResNet-18 ---
    print("\n" + "-" * 60)
    print("1/3. Evaluating ResNet-18...")
    print("-" * 60, flush=True)
    r18 = models.resnet18(weights=None)
    num_ftrs = r18.fc.in_features
    r18.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(classes))
    )
    r18.load_state_dict(torch.load(mels_dir / "resnet18.pth", map_location="cpu", weights_only=False))
    r18.to(device)
    p_r18, r18_seg, r18_trk = eval_model(r18, "ResNet-18")
    np.save(mels_dir / "resnet18_test_probs.npy", p_r18)

    # --- 2. EfficientNet-B0 ---
    print("\n" + "-" * 60)
    print("2/3. Evaluating EfficientNet-B0...")
    print("-" * 60, flush=True)
    eff = models.efficientnet_b0(weights=None)
    in_features = eff.classifier[1].in_features
    eff.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(classes))
    )
    eff.load_state_dict(torch.load(mels_dir / "efficientnetb0.pth", map_location="cpu", weights_only=False))
    eff.to(device)
    p_eff, eff_seg, eff_trk = eval_model(eff, "EfficientNet-B0")
    np.save(mels_dir / "efficientnet_test_probs.npy", p_eff)

    # --- 3. MobileNetV2 ---
    print("\n" + "-" * 60)
    print("3/3. Evaluating MobileNetV2...")
    print("-" * 60, flush=True)
    mob = models.mobilenet_v2(weights=None)
    in_features = mob.classifier[1].in_features
    mob.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(classes))
    )
    mob.load_state_dict(torch.load(mels_dir / "mobilenetv2.pth", map_location="cpu", weights_only=False))
    mob.to(device)
    p_mob, mob_seg, mob_trk = eval_model(mob, "MobileNetV2")
    np.save(mels_dir / "mobilenetv2_test_probs.npy", p_mob)

    # Save shared metadata for ensemble
    np.save(mels_dir / "resnet18_test_labels.npy", y_test)
    np.save(mels_dir / "resnet18_test_tracks.npy", test_tracks)
    np.save(mels_dir / "resnet18_classes.npy", np.array(classes))

    print("\n" + "=" * 70)
    print("CNN EVALUATION COMPLETE")
    print(f"Elapsed: {time.time() - start_time:.1f}s")
    print("=" * 70, flush=True)

if __name__ == '__main__':
    main()
