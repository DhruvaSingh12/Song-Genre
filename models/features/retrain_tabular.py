import os
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb

def main():
    start_time = time.time()
    base_dir = Path(__file__).resolve().parents[2]
    features_dir = base_dir / "models" / "features"
    backup_dir = features_dir / "checkpoints_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STARTING RETRAINING OF TABULAR ACOUSTIC MODELS")
    print("=" * 70)

    # 1. Backup old models if present
    for fname in ["linearsvm.joblib", "randomforest.joblib", "mlp.pt"]:
        src = features_dir / fname
        dst = backup_dir / fname
        if src.exists() and not dst.exists():
            print(f"Backing up original checkpoint: {fname} -> checkpoints_backup/")
            shutil.copy2(src, dst)

    # 2. Locate Data
    data_dir = base_dir / "csvs" / "CSVs"
    if not (data_dir / "train.csv").exists():
        data_dir = base_dir / "csvs"
    if not (data_dir / "train.csv").exists():
        data_dir = base_dir

    print(f"\nLoading CSV datasets from: {data_dir}")
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "valid.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    TARGET_COL = "Genre"
    IGNORE_COLS = [TARGET_COL, "Filename"]
    feature_cols = [c for c in train_df.columns if c not in IGNORE_COLS]

    print(f"Dataset summary: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Total features: {len(feature_cols)}")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]

    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL]

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    test_filenames = test_df["Filename"].tolist()
    test_tracks = [os.path.splitext(fn)[0].split("_seg")[0] for fn in test_filenames]

    classes = sorted(y_train.unique())
    print(f"Classes ({len(classes)}): {classes}")

    results = {}
    probs = {}

    # MODEL 1: RBF KERNEL SUPPORT VECTOR MACHINE
    print("\n" + "-" * 60)
    print("1/4. Retraining RBF Kernel Support Vector Machine (C=3.0, balanced)...")
    print("-" * 60)
    svm_start = time.time()
    svm_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=3.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42
        ))
    ])
    svm_pipeline.fit(X_train, y_train)

    val_preds_svm = svm_pipeline.predict(X_val)
    val_probs_svm = svm_pipeline.predict_proba(X_val)
    test_preds_svm = svm_pipeline.predict(X_test)
    test_probs_svm = svm_pipeline.predict_proba(X_test)

    val_acc_svm = accuracy_score(y_val, val_preds_svm)
    test_acc_svm = accuracy_score(y_test, test_preds_svm)
    test_f1_svm = f1_score(y_test, test_preds_svm, average="weighted")
    print(f"  -> SVM Val Accuracy : {val_acc_svm * 100:.2f}%")
    print(f"  -> SVM Test Accuracy: {test_acc_svm * 100:.2f}% (Weighted F1: {test_f1_svm:.4f})")
    print(f"  -> Training time    : {time.time() - svm_start:.1f}s")

    joblib.dump(svm_pipeline, features_dir / "linearsvm.joblib")
    np.save(features_dir / "svm_val_probs.npy", val_probs_svm)
    np.save(features_dir / "svm_test_probs.npy", test_probs_svm)
    results["SVM (RBF Kernel)"] = (val_acc_svm, test_acc_svm)
    probs["svm"] = test_probs_svm

    # MODEL 2: REGULARIZED RANDOM FOREST
    print("\n" + "-" * 60)
    print("2/4. Retraining Regularized Random Forest (max_depth=16, min_samples_leaf=3)...")
    print("-" * 60)
    rf_start = time.time()
    rf_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            max_depth=16,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ))
    ])
    rf_pipeline.fit(X_train, y_train)

    val_preds_rf = rf_pipeline.predict(X_val)
    val_probs_rf = rf_pipeline.predict_proba(X_val)
    test_preds_rf = rf_pipeline.predict(X_test)
    test_probs_rf = rf_pipeline.predict_proba(X_test)

    val_acc_rf = accuracy_score(y_val, val_preds_rf)
    test_acc_rf = accuracy_score(y_test, test_preds_rf)
    test_f1_rf = f1_score(y_test, test_preds_rf, average="weighted")
    print(f"  -> Random Forest Val Accuracy : {val_acc_rf * 100:.2f}%")
    print(f"  -> Random Forest Test Accuracy: {test_acc_rf * 100:.2f}% (Weighted F1: {test_f1_rf:.4f})")
    print(f"  -> Training time              : {time.time() - rf_start:.1f}s")

    joblib.dump(rf_pipeline, features_dir / "randomforest.joblib")
    np.save(features_dir / "rf_val_probs.npy", val_probs_rf)
    np.save(features_dir / "rf_test_probs.npy", test_probs_rf)
    results["Random Forest (Regularized)"] = (val_acc_rf, test_acc_rf)
    probs["rf"] = test_probs_rf

    # MODEL 3: LIGHTGBM GRADIENT BOOSTING
    print("\n" + "-" * 60)
    print("3/4. Training LightGBM Gradient Boosted Trees...")
    print("-" * 60)
    lgb_start = time.time()
    lgb_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("lgb", lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ))
    ])
    lgb_pipeline.fit(X_train, y_train)

    val_preds_lgb = lgb_pipeline.predict(X_val)
    val_probs_lgb = lgb_pipeline.predict_proba(X_val)
    test_preds_lgb = lgb_pipeline.predict(X_test)
    test_probs_lgb = lgb_pipeline.predict_proba(X_test)

    val_acc_lgb = accuracy_score(y_val, val_preds_lgb)
    test_acc_lgb = accuracy_score(y_test, test_preds_lgb)
    test_f1_lgb = f1_score(y_test, test_preds_lgb, average="weighted")
    print(f"  -> LightGBM Val Accuracy : {val_acc_lgb * 100:.2f}%")
    print(f"  -> LightGBM Test Accuracy: {test_acc_lgb * 100:.2f}% (Weighted F1: {test_f1_lgb:.4f})")
    print(f"  -> Training time         : {time.time() - lgb_start:.1f}s")

    joblib.dump(lgb_pipeline, features_dir / "lightgbm.joblib")
    np.save(features_dir / "lgb_val_probs.npy", val_probs_lgb)
    np.save(features_dir / "lgb_test_probs.npy", test_probs_lgb)
    results["LightGBM"] = (val_acc_lgb, test_acc_lgb)
    probs["lgb"] = test_probs_lgb

    # MODEL 4: PYTORCH MLP (BATCHNORM, MISH, ADAMW, COSINE SCHEDULER)
    print("\n" + "-" * 60)
    print("4/4. Retraining PyTorch MLP (BatchNorm, Mish, AdamW, CosineLR on GPU)...")
    print("-" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  -> Training on device: {device}")

    mlp_start = time.time()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_ts_imp = imputer.transform(X_test)

    X_tr_sc = scaler.fit_transform(X_tr_imp).astype(np.float32)
    X_val_sc = scaler.transform(X_val_imp).astype(np.float32)
    X_ts_sc = scaler.transform(X_ts_imp).astype(np.float32)

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_train).astype(np.int64)
    y_val_enc = le.transform(y_val).astype(np.int64)
    y_ts_enc = le.transform(y_test).astype(np.int64)

    train_ds = TensorDataset(torch.from_numpy(X_tr_sc), torch.from_numpy(y_tr_enc))
    val_ds = TensorDataset(torch.from_numpy(X_val_sc), torch.from_numpy(y_val_enc))
    test_ds = TensorDataset(torch.from_numpy(X_ts_sc), torch.from_numpy(y_ts_enc))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    class ImprovedMLP(nn.Module):
        def __init__(self, input_dim=31, num_classes=9, dropout_rate=0.35):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.Mish(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.Mish(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.Mish(),
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    mlp_model = ImprovedMLP(input_dim=len(feature_cols), num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=1e-3, weight_decay=1e-3)
    epochs = 40
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    def eval_mlp(model, loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / total

    best_val_acc = 0.0
    best_weights = None

    for ep in range(1, epochs + 1):
        mlp_model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(mlp_model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_acc = eval_mlp(mlp_model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in mlp_model.state_dict().items()}

    mlp_model.load_state_dict(best_weights)
    torch.save(mlp_model.state_dict(), features_dir / "mlp.pt")

    def get_probs_and_preds(model, loader):
        model.eval()
        probs_list, preds_list = [], []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                out = model(xb)
                p = F.softmax(out, dim=1).cpu().numpy()
                probs_list.append(p)
                preds_list.append(p.argmax(axis=1))
        return np.vstack(probs_list), np.concatenate(preds_list)

    val_probs_mlp, val_preds_mlp = get_probs_and_preds(mlp_model, val_loader)
    test_probs_mlp, test_preds_mlp = get_probs_and_preds(mlp_model, test_loader)

    val_acc_mlp = accuracy_score(y_val_enc, val_preds_mlp)
    test_acc_mlp = accuracy_score(y_ts_enc, test_preds_mlp)
    test_f1_mlp = f1_score(y_ts_enc, test_preds_mlp, average="weighted")
    print(f"  -> PyTorch MLP Val Accuracy : {val_acc_mlp * 100:.2f}%")
    print(f"  -> PyTorch MLP Test Accuracy: {test_acc_mlp * 100:.2f}% (Weighted F1: {test_f1_mlp:.4f})")
    print(f"  -> Training time            : {time.time() - mlp_start:.1f}s")

    np.save(features_dir / "mlp_val_probs.npy", val_probs_mlp)
    np.save(features_dir / "mlp_test_probs.npy", test_probs_mlp)
    results["PyTorch MLP"] = (val_acc_mlp, test_acc_mlp)
    probs["mlp"] = test_probs_mlp

    # Save classes
    np.save(features_dir / "classes.npy", np.array(classes))

    # 5. TABULAR MULTI-MODEL ENSEMBLE
    print("\n" + "=" * 70)
    print("5. COMPUTING TABULAR MULTI-MODEL ENSEMBLE")
    print("=" * 70)

    # Weighted Soft Voting: LGBM(0.35) + RF(0.25) + SVM(0.25) + MLP(0.15)
    p_tab_ensemble = (
        0.35 * probs["lgb"] +
        0.25 * probs["rf"] +
        0.25 * probs["svm"] +
        0.15 * probs["mlp"]
    )
    seg_preds_ensemble = np.array(classes)[p_tab_ensemble.argmax(axis=1)]
    seg_acc_ensemble = accuracy_score(y_test, seg_preds_ensemble)
    seg_f1_ensemble = f1_score(y_test, seg_preds_ensemble, average="weighted")

    print(f"\n>> Tabular Ensemble Segment-Level Test Accuracy: {seg_acc_ensemble * 100:.2f}% (F1: {seg_f1_ensemble:.4f})")

    # Track-Level (Full Song) Aggregation
    prob_cols = list(range(len(classes)))
    df_track = pd.DataFrame(p_tab_ensemble, columns=prob_cols)
    df_track["track_id"] = test_tracks.values if hasattr(test_tracks, "values") else test_tracks
    df_track["true_label"] = y_test.values

    track_probs = df_track.groupby("track_id")[prob_cols].mean()
    track_y_true = df_track.groupby("track_id")["true_label"].first()
    track_preds = np.array(classes)[track_probs.values.argmax(axis=1)]
    track_acc = accuracy_score(track_y_true, track_preds)
    track_f1 = f1_score(track_y_true, track_preds, average="weighted")

    print(f">> Tabular Ensemble Track-Level (Full Song) Test Accuracy: {track_acc * 100:.2f}% (F1: {track_f1:.4f})")

    # Summary table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: RETRAINED TABULAR MODELS BENCHMARK")
    print("=" * 70)
    print(f"{'Model Name':<30} | {'Val Accuracy':<15} | {'Test Accuracy':<15}")
    print("-" * 66)
    for name, (va, ta) in results.items():
        print(f"{name:<30} | {va * 100:>13.2f}% | {ta * 100:>13.2f}%")
    print("-" * 66)
    print(f"{'Tabular Ensemble (Segment)':<30} | {'-':>14} | {seg_acc_ensemble * 100:>13.2f}%")
    print(f"{'Tabular Ensemble (Full Song)':<30} | {'-':>14} | {track_acc * 100:>13.2f}%")
    print("=" * 70)
    print(f"Total time elapsed: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
