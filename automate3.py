import os
import csv
from typing import List
import numpy as np
import librosa
from tqdm import tqdm

TRIMMED_WAV_ROOT = os.path.join(os.path.dirname(__file__), "trimmed_wavs")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "merged_features_generated.csv")

# Librosa / feature extraction settings
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 13


def energy_entropy(frame: np.ndarray, n_slices: int = 10) -> float:
    # Short-term energy entropy for a frame
    frame = frame.astype(float)
    total_energy = np.sum(frame ** 2)
    if total_energy <= 0:
        return 0.0

    slice_len = max(1, len(frame) // n_slices)
    energies = []
    for i in range(n_slices):
        start = i * slice_len
        end = start + slice_len
        if start >= len(frame):
            break
        energies.append(np.sum(frame[start:end] ** 2))

    energies = np.array(energies, dtype=float)
    probs = energies / (energies.sum() + 1e-12)
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def amplitude_envelope_mean(y: np.ndarray, frame_size: int = 1024, hop: int = 512) -> float:
    # Mean amplitude envelope (RMS-like, but via per-frame max)
    env = []
    for i in range(0, len(y), hop):
        frame = y[i : i + frame_size]
        if len(frame) == 0:
            continue
        env.append(np.max(np.abs(frame)))
    if not env:
        return 0.0
    return float(np.mean(env))


def spectral_slope(magnitude: np.ndarray, freqs: np.ndarray) -> float:
    # Simple spectral slope over frequency bins
    if magnitude.ndim > 1:
        mag = magnitude.mean(axis=1)
    else:
        mag = magnitude
    mag = mag + 1e-12
    x = freqs
    y = 20 * np.log10(mag)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    x_mean = x.mean()
    y_mean = y.mean()
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2) + 1e-12
    return float(num / den)


def spectral_moments(magnitude: np.ndarray, freqs: np.ndarray) -> tuple[float, float]:
    # Spectral skewness and kurtosis from magnitude spectrum
    if magnitude.ndim > 1:
        mag = magnitude.mean(axis=1)
    else:
        mag = magnitude
    mag = mag + 1e-12
    mag_norm = mag / (mag.sum() + 1e-12)
    mu = np.sum(freqs * mag_norm)
    sigma = np.sqrt(np.sum(((freqs - mu) ** 2) * mag_norm) + 1e-12)
    skew = np.sum(((freqs - mu) / sigma) ** 3 * mag_norm)
    kurt = np.sum(((freqs - mu) / sigma) ** 4 * mag_norm)
    return float(skew), float(kurt)


def extract_features_for_file(path: str) -> List[float]:
    y, sr = librosa.load(path, sr=SR, mono=True)

    # Skip completely empty or extremely short signals
    if y.size == 0 or y.size < N_FFT:
        raise ValueError("Audio too short or empty for reliable feature extraction")

    # Basic amplitude statistics (guard against NaNs on edge cases)
    mean_amp = float(np.mean(y)) if y.size > 0 else 0.0
    var_amp = float(np.var(y)) if y.size > 1 else 0.0
    if y.size > 1 and np.std(y) > 0:
        std_y = np.std(y) + 1e-12
        var_y = np.var(y) + 1e-12
        skew_amp = float(((y - mean_amp) ** 3).mean() / (std_y ** 3))
        kurt_amp = float(((y - mean_amp) ** 4).mean() / (var_y ** 2))
    else:
        skew_amp = 0.0
        kurt_amp = 0.0
    ptp_amp = float(np.ptp(y)) if y.size > 0 else 0.0

    # Frame-based features
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)))
    rms = float(np.mean(librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)))
    amp_env_mean = amplitude_envelope_mean(y)
    ent = energy_entropy(y)

    # Spectral features
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    spec_centroid = float(np.mean(librosa.feature.spectral_centroid(S=S, sr=sr)))
    spec_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr)))
    spec_contrast = float(np.mean(librosa.feature.spectral_contrast(S=S, sr=sr)))
    spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr)))
    spec_flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    spec_flux = float(np.mean(librosa.onset.onset_strength(S=S)))

    spec_skew, spec_kurt = spectral_moments(S, freqs)
    spec_slope = spectral_slope(S, freqs)

    # MFCCs (first 13, mean over time)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_means = [float(np.mean(mfcc[i])) for i in range(N_MFCC)]

    return [ zcr, rms, amp_env_mean, ent, mean_amp, var_amp, skew_amp, kurt_amp, ptp_amp,
        spec_centroid, spec_bandwidth, spec_contrast, spec_rolloff, spec_flatness,
        spec_flux, spec_skew, spec_kurt, spec_slope, *mfcc_means ]

HEADERS = [
    "Filename",
    "Zero_Crossing_Rate",
    "RMS_Energy",
    "Amplitude_Envelope_Mean",
    "Energy_Entropy",
    "Mean_Amplitude",
    "Variance_Amplitude",
    "Skewness_Amplitude",
    "Kurtosis_Amplitude",
    "Peak_to_Peak_Amplitude",
    "Spectral_Centroid",
    "Spectral_Bandwidth",
    "Spectral_Contrast",
    "Spectral_Rolloff",
    "Spectral_Flatness",
    "Spectral_Flux",
    "Spectral_Skewness",
    "Spectral_Kurtosis",
    "Spectral_Slope",
] + [f"MFCC_{i}" for i in range(1, N_MFCC + 1)] + ["Genre"]


def main() -> None:
    rows = []

    # Expect structure trimmed_wavs/<genre>/*.wav
    for genre in sorted(os.listdir(TRIMMED_WAV_ROOT)):
        genre_dir = os.path.join(TRIMMED_WAV_ROOT, genre)
        if not os.path.isdir(genre_dir):
            continue
        files = [f for f in sorted(os.listdir(genre_dir)) if f.lower().endswith(".wav")]
        print(f"Processing genre '{genre}' with {len(files)} files...")
        for fname in tqdm(files, desc=f"{genre}"):
            path = os.path.join(genre_dir, fname)
            try:
                feats = extract_features_for_file(path)
                rows.append([fname, *feats, genre])
            except Exception as e:
                print(f"Error processing {path}: {e}")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()