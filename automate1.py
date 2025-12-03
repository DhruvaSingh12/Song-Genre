import os
import glob
from typing import Any
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from moviepy import AudioFileClip
from yt_dlp import YoutubeDL
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LINKS_FOLDER = os.path.join(BASE_DIR, "links")
DOWNLOAD_FOLDER = os.path.join(BASE_DIR, "downloads")
TRIMMED_FOLDER = os.path.join(BASE_DIR, "trimmed_wavs")
SPECTROGRAM_FOLDER = os.path.join(BASE_DIR, "mel_spectrograms")

# Ensure output directories exist
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(TRIMMED_FOLDER, exist_ok=True)
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)

def download_songs():
    csv_files = glob.glob(os.path.join(LINKS_FOLDER, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in the links folder.")
        return

    print(f"Found {len(csv_files)} CSV files. Starting download process...")

    for csv_file in csv_files:
        genre = os.path.splitext(os.path.basename(csv_file))[0]
        print(f"\nProcessing genre: {genre}")
        
        # Create genre specific download folder
        genre_download_folder = os.path.join(DOWNLOAD_FOLDER, genre)
        os.makedirs(genre_download_folder, exist_ok=True)
        
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

        if 'URL' not in df.columns:
            print(f"Skipping {csv_file}: 'URL' column not found.")
            continue

        for idx, (_, row) in enumerate(df.iterrows()):
            url = row['URL']
            if not isinstance(url, str) or not url.strip():
                continue
            
            # Naming convention: genre_index (e.g., hiphop_1)
            filename_base = f"{genre}_{idx+1}"
            
            # Check if file already exists (to avoid re-downloading)
            existing_files = glob.glob(os.path.join(genre_download_folder, f"{filename_base}.*"))
            if existing_files:
                print(f"Skipping {filename_base}, already exists.")
                continue

            print(f"Downloading {idx+1}/{len(df)}: {url}")
            
            ydl_opts: Any = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(genre_download_folder, f'{filename_base}.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': True,
            }

            try:
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            except Exception as e:
                print(f"Failed to download {url}: {e}")

def trim_audio_segments():
    print("\nStarting audio trimming...")
    
    # Define the 10 segments (start, end) in seconds
    segment_times = [(10, 20), (30, 40), (50, 60), (70, 80), (90, 100), (110, 120), (130, 140), (150, 160), (170, 180), (190, 200)]

    # Iterate over genre folders in DOWNLOAD_FOLDER
    for genre in os.listdir(DOWNLOAD_FOLDER):
        genre_path = os.path.join(DOWNLOAD_FOLDER, genre)
        if not os.path.isdir(genre_path):
            continue
            
        # Create genre specific trimmed folder
        genre_trimmed_folder = os.path.join(TRIMMED_FOLDER, genre)
        os.makedirs(genre_trimmed_folder, exist_ok=True)

        audio_files = [f for f in os.listdir(genre_path) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.webm', '.flac'))]
        
        for filename in audio_files:
            input_path = os.path.join(genre_path, filename)
            base_name = os.path.splitext(filename)[0]
            
            print(f"Processing: {filename}")

            try:
                # Load audio file
                audio: Any = AudioFileClip(input_path)
                duration = audio.duration
                
                for i, (start, end) in enumerate(segment_times, start=1):
                    # Check if the song is long enough for this segment
                    if duration >= end:
                        output_filename = f"{base_name}_seg{i}.wav"
                        output_path = os.path.join(genre_trimmed_folder, output_filename)
                        
                        if os.path.exists(output_path):
                            continue

                        try:
                            # MoviePy's Clip API uses 'subclipped' for current version
                            segment_clip = audio.subclipped(start, end)
                            # Write as wav
                            segment_clip.write_audiofile(output_path, codec='pcm_s16le', logger=None)

                        except Exception as e:
                            print(f"  Error creating segment {i}: {e}")
                    else:
                        # If song is too short for this segment, we stop trying further segments
                        pass
                
                audio.close()

            except Exception as e:
                print(f"Error processing {filename}: {e}")

def create_spectrogram(args):
    filename, wav_path, spec_path = args

    try:
        y, sr = librosa.load(wav_path)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Higher quality and with scales
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, vmin=-80, vmax=0)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel-frequency spectrogram of {filename}')
        plt.tight_layout()
        plt.savefig(spec_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Failed to generate spectrogram for {filename}: {e}")

def generate_mel_spectrograms():
    print("\nGenerating Mel Spectrograms...")
    
    tasks = []

    # Iterate over genre folders in TRIMMED_FOLDER
    for genre in os.listdir(TRIMMED_FOLDER):
        genre_path = os.path.join(TRIMMED_FOLDER, genre)
        if not os.path.isdir(genre_path):
            continue

        # Create genre specific spectrogram folder
        genre_spectrogram_folder = os.path.join(SPECTROGRAM_FOLDER, genre)
        os.makedirs(genre_spectrogram_folder, exist_ok=True)

        wav_files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
        
        for filename in wav_files:
            wav_path = os.path.join(genre_path, filename)
            spec_filename = f"{os.path.splitext(filename)[0]}.png"
            spec_path = os.path.join(genre_spectrogram_folder, spec_filename)
            tasks.append((filename, wav_path, spec_path))

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(create_spectrogram, tasks), total=len(tasks), desc="Processing Spectrograms"))


if __name__ == "__main__":
    download_songs()
    trim_audio_segments()
    generate_mel_spectrograms()
    print("\nAll tasks completed successfully!")