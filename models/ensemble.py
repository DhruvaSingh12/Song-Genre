import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    base_dir = Path(__file__).resolve().parent
    features_dir = base_dir / "features"
    mels_dir = base_dir / "mels"

    print("=" * 60)
    print("SONG GENRE MULTI-MODAL ENSEMBLE & TRACK VOTING")
    print("=" * 60)

    model_probs = {}

    # Check for CNN probability distributions
    for cnn_name in ["resnet18", "efficientnet", "mobilenetv2"]:
        path = mels_dir / f"{cnn_name}_test_probs.npy"
        if not path.exists():
            path = base_dir / f"{cnn_name}_test_probs.npy"
        if path.exists():
            model_probs[cnn_name] = np.load(path)
            print(f"Loaded {cnn_name} test probabilities: shape {model_probs[cnn_name].shape}")

    # Check for Tabular probability distributions
    for tab_name in ["mlp", "svm", "lgb", "rf"]:
        path = features_dir / f"{tab_name}_test_probs.npy"
        if path.exists():
            model_probs[tab_name] = np.load(path)
            print(f"Loaded {tab_name} test probabilities: shape {model_probs[tab_name].shape}")

    if not model_probs:
        print("\nNo exported probability files (*_test_probs.npy) found.")
        print("Please run the updated evaluation cells in your training notebooks to generate them.")
        return

    # Load ground truth labels and track IDs
    labels_path = mels_dir / "resnet18_test_labels.npy"
    if not labels_path.exists():
        labels_path = base_dir / "resnet18_test_labels.npy"

    tracks_path = mels_dir / "resnet18_test_tracks.npy"
    if not tracks_path.exists():
        tracks_path = base_dir / "resnet18_test_tracks.npy"

    classes_path = mels_dir / "resnet18_classes.npy"
    if not classes_path.exists():
        classes_path = features_dir / "classes.npy"

    y_true = np.load(labels_path, allow_pickle=True) if labels_path.exists() else None
    track_ids = np.load(tracks_path, allow_pickle=True) if tracks_path.exists() else None
    classes = np.load(classes_path, allow_pickle=True) if classes_path.exists() else None

    # Fallback to tabular CSV metadata if CNN test arrays not yet exported
    test_csv_path = base_dir.parent / "csvs" / "test.csv"
    if not test_csv_path.exists():
        test_csv_path = base_dir.parent / "csvs" / "CSVs" / "test.csv"
    if test_csv_path.exists():
        df_csv = pd.read_csv(test_csv_path)
        if classes is None and "Genre" in df_csv.columns:
            classes = np.array(sorted(df_csv["Genre"].unique()))
        if y_true is None and "Genre" in df_csv.columns:
            genre_to_idx = {g: i for i, g in enumerate(classes)}
            y_true = df_csv["Genre"].map(genre_to_idx).values
        if track_ids is None and "Filename" in df_csv.columns:
            track_ids = df_csv["Filename"].apply(
                lambda x: str(x).split("_seg")[0] if "_seg" in str(x) else str(x)
            ).values

    # Default heuristic weights prioritizing strongest models
    default_weights = {
        "resnet18": 0.35,
        "efficientnet": 0.30,
        "mobilenetv2": 0.15,
        "mlp": 0.08,
        "svm": 0.06,
        "lgb": 0.04,
        "rf": 0.02,
    }

    available_models = list(model_probs.keys())
    raw_weights = [default_weights.get(m, 0.1) for m in available_models]
    weights = np.array(raw_weights) / sum(raw_weights)

    print("\nModel Weights in Ensemble:")
    for m, w in zip(available_models, weights):
        print(f"  * {m:15s}: {w * 100:.1f}%")

    # Compute weighted ensemble
    p_ensemble = np.zeros_like(next(iter(model_probs.values())))
    for m, w in zip(available_models, weights):
        p_ensemble += w * model_probs[m]

    seg_preds = np.argmax(p_ensemble, axis=1)

    if y_true is not None:
        seg_acc = accuracy_score(y_true, seg_preds)
        print("\n" + "=" * 50)
        print(f"Segment-Level Ensemble Accuracy: {seg_acc * 100:.2f}%")
        print("=" * 50)

        # Track-Level (Full Song) Evaluation
        if track_ids is not None:
            num_classes = p_ensemble.shape[1]
            prob_cols = list(range(num_classes))
            df_ensemble = pd.DataFrame(p_ensemble, columns=prob_cols)
            df_ensemble["track_id"] = track_ids
            df_ensemble["true_label"] = y_true

            # Average probabilities across all 10-second segments of each song
            track_probs = df_ensemble.groupby("track_id")[prob_cols].mean()
            track_y_true = df_ensemble.groupby("track_id")["true_label"].first()
            track_preds = track_probs.values.argmax(axis=1)
            track_acc = accuracy_score(track_y_true, track_preds)

            print(f"Track-Level (Full Song) Ensemble Accuracy: {track_acc * 100:.2f}%")
            print("=" * 50)

        if classes is not None:
            print("\nClassification Report (Segment-Level):")
            print(classification_report(y_true, seg_preds, target_names=[str(c) for c in classes]))

if __name__ == "__main__":
    main()
