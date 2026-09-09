#!/usr/bin/env python3
"""
Generate SCP file from folder with iterative WAV files.

Finds all WAV files matching a specific iteration number in subdirectories.
File structure: folder/audio_id/iter_N.wav
SCP format: audio_id /absolute/path/to/iter_N.wav
"""

import os
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate SCP file from iterative WAV files")
    parser.add_argument("--folder", help="Input folder path (e.g., data/LibriTTS/test_clean/CosyVoice)")
    parser.add_argument("--iter", type=int, help="Iteration number (e.g., 1 for iter_1.wav)")
    args = parser.parse_args()

    # Convert folder to absolute path
    folder_path = Path(args.folder).resolve()
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Generate output filename
    output_filename = f"iter_{args.iter}.scp"
    output_path = folder_path / output_filename
    print(f"Generating SCP file: {output_path}")
    
    # Collect all matching WAV files
    results = []
    
    for audio_dir in sorted(folder_path.iterdir()):
        if not audio_dir.is_dir():
            continue
        
        # Look for iter_N.wav file
        wav_file = audio_dir / f"iter_{args.iter}.wav"
        
        if wav_file.exists():
            audio_id = audio_dir.name
            absolute_wav_path = wav_file.resolve()
            results.append((audio_id, str(absolute_wav_path)))
    
    # Write to SCP file
    with open(output_path, "w", encoding="utf-8") as f_out:
        for audio_id, wav_path in results:
            f_out.write(f"{audio_id} {wav_path}\n")
    
    print(f"Generated SCP file: {output_path}")
    print(f"Total entries: {len(results)}")


if __name__ == "__main__":
    main()
