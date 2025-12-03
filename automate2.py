import os
import numpy as np
import librosa
from moviepy import AudioFileClip
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_FOLDER = os.path.join(BASE_DIR, "downloads")
TRIMMED_FOLDER = os.path.join(BASE_DIR, "trimmed_wavs")
MEL_ARRAY_FOLDER = os.path.join(BASE_DIR, "mel_arrays")

# Ensure output directories exist
os.makedirs(TRIMMED_FOLDER, exist_ok=True)
os.makedirs(MEL_ARRAY_FOLDER, exist_ok=True)

# Define the 10 segments (start, end) in seconds
SEGMENT_TIMES = [
    (10, 20), (30, 40), (50, 60), (70, 80), (90, 100),
    (110, 120), (130, 140), (150, 160), (170, 180), (190, 200)
]

def trim_single_file(args):
    input_path, genre_trimmed_folder, filename = args
    base_name = os.path.splitext(filename)[0]
    
    try:
        # Load audio file
        audio = AudioFileClip(input_path)
        duration = audio.duration
        
        for i, (start, end) in enumerate(SEGMENT_TIMES, start=1):
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
                    print(f"  Error creating segment {i} for {filename}: {e}")
            else:
                # If song is too short for this segment, we stop trying further segments
                pass
        
        audio.close()

    except Exception as e:
        print(f"Error processing {filename}: {e}")

def trim_audio_segments():
    print("\nStarting audio trimming...")
    
    tasks = []

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
            tasks.append((input_path, genre_trimmed_folder, filename))

    if tasks:
        # Use ProcessPoolExecutor to parallelize
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(trim_single_file, tasks), total=len(tasks), desc="Trimming Audio Files"))
    else:
        print("No audio files found to trim.")

def create_mel_array(args):
    filename, wav_path, array_path = args
    
    if os.path.exists(array_path):
        return

    try:
        y, sr = librosa.load(wav_path)
        # Generate Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        # Convert to dB (Log-Mel Spectrogram)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Save as numpy array
        np.save(array_path, S_dB)
        
    except Exception as e:
        print(f"Failed to generate mel array for {filename}: {e}")

def generate_mel_arrays():
    print("\nGenerating Mel Arrays...")
    
    tasks = []

    # Iterate over genre folders in TRIMMED_FOLDER
    for genre in os.listdir(TRIMMED_FOLDER):
        genre_path = os.path.join(TRIMMED_FOLDER, genre)
        if not os.path.isdir(genre_path):
            continue

        # Create genre specific array folder
        genre_array_folder = os.path.join(MEL_ARRAY_FOLDER, genre)
        os.makedirs(genre_array_folder, exist_ok=True)

        wav_files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
        
        for filename in wav_files:
            wav_path = os.path.join(genre_path, filename)
            array_filename = f"{os.path.splitext(filename)[0]}.npy"
            array_path = os.path.join(genre_array_folder, array_filename)
            tasks.append((filename, wav_path, array_path))

    # Use ProcessPoolExecutor to parallelize
    if tasks:
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(create_mel_array, tasks), total=len(tasks), desc="Processing Mel Arrays"))
    else:
        print("No WAV files found to process.")

if __name__ == "__main__":
    trim_audio_segments()
    generate_mel_arrays()
    print("\nAll tasks completed successfully!")