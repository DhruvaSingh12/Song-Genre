import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(__file__)
FEATURE_CSV = os.path.join(BASE_DIR, "features.csv")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "valid.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")


def main() -> None:
    if not os.path.exists(FEATURE_CSV):
        raise FileNotFoundError(f"Feature CSV not found at {FEATURE_CSV}")

    df = pd.read_csv(FEATURE_CSV)

    if "Filename" not in df.columns or "Genre" not in df.columns:
        raise ValueError("CSV must contain 'Filename' and 'Genre' columns")

    # Derive track_id from filename convention e.g. 'rock_100_3.wav' or 'rock_100_seg3.wav'
    def get_track_id(fn: str) -> str:
        name = os.path.splitext(fn)[0]
        # If we have _segX, strip that part: rock_100_seg3 -> rock_100
        if "_seg" in name:
            return name.split("_seg")[0]
        # Else, strip the last underscore chunk: rock_100_3 -> rock_100
        parts = name.split("_")
        if len(parts) >= 3 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return name

    df["track_id"] = df["Filename"].astype(str).apply(get_track_id)

    # Build track-level label (assume all segments of a track share Genre)
    track_to_label = (
        df.groupby("track_id")["Genre"].agg(lambda x: x.iloc[0]).to_dict()
    )

    unique_tracks = np.array(list(track_to_label.keys()))
    unique_labels = np.array([track_to_label[tid] for tid in unique_tracks])

    # Map labels to integers for stratified splitting
    _, label_indices = np.unique(unique_labels, return_inverse=True)

    # 75% train, 15% val, 10% test (via 75/25 then 60/40)
    train_tracks, temp_tracks, y_train_tracks, y_temp_tracks = train_test_split(
        unique_tracks,
        label_indices,
        test_size=0.25,
        random_state=42,
        stratify=label_indices,
    )

    val_tracks, test_tracks, y_val_tracks, y_test_tracks = train_test_split(
        temp_tracks,
        y_temp_tracks,
        test_size=0.4,
        random_state=42,
        stratify=y_temp_tracks,
    )

    train_ids_set = set(train_tracks)
    val_ids_set = set(val_tracks)
    test_ids_set = set(test_tracks)

    # Sanity checks (no overlap)
    assert train_ids_set.isdisjoint(val_ids_set)
    assert train_ids_set.isdisjoint(test_ids_set)
    assert val_ids_set.isdisjoint(test_ids_set)

    print(
        f"Tracks split: Train={len(train_tracks)}, Val={len(val_tracks)}, Test={len(test_tracks)}"
    )

    # Assign rows based on track_id membership
    train_df = df[df["track_id"].isin(train_ids_set)].copy()
    val_df = df[df["track_id"].isin(val_ids_set)].copy()
    test_df = df[df["track_id"].isin(test_ids_set)].copy()

    # Drop helper column in outputs
    train_df = train_df.drop(columns=["track_id"])
    val_df = val_df.drop(columns=["track_id"])
    test_df = test_df.drop(columns=["track_id"])

    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"Saved train split to {TRAIN_CSV} ({len(train_df)} rows)")
    print(f"Saved val split to {VAL_CSV} ({len(val_df)} rows)")
    print(f"Saved test split to {TEST_CSV} ({len(test_df)} rows)")

if __name__ == "__main__":
    main()