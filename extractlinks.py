import os
import csv
from typing import Any
from yt_dlp import YoutubeDL

url = "https://www.youtube.com/playlist?list=PLO_1AmtK1TMRi01-V_tdHKDLBu7cNWIgf"

output_dir = os.path.join(os.path.dirname(__file__), 'links')
os.makedirs(output_dir, exist_ok=True)
csv_file = os.path.join(output_dir, 'jazz.csv')

with YoutubeDL() as ydl:
    info: Any = ydl.extract_info(url, download=False)
if info and 'entries' in info:
    print(f"Saving song list to {csv_file}...")
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Title', 'URL'])
        for entry in info['entries']:
            title = entry.get('title', 'Unknown')
            video_url = entry.get('webpage_url') or entry.get('url')
            if not video_url and entry.get('id'):
                 video_url = f"https://www.youtube.com/watch?v={entry.get('id')}"
            writer.writerow([title, video_url])