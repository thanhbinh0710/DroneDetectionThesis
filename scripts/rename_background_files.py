#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to rename newly added background audio files to BACKGROUND_XXX.wav format
Usage: python scripts/rename_background_files.py
"""

import os
import sys
import glob
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)


def main():
    """Rename background files to BACKGROUND_XXX.wav format"""
    
    background_dir = os.path.join(project_root, "data", "raw", "background")
    
    print("="*70)
    print("BACKGROUND AUDIO FILES - BATCH RENAME")
    print("="*70)
    
    # Check if background folder exists
    if not os.path.exists(background_dir):
        print(f"\nError: Background folder not found: {background_dir}")
        return
    
    # Find existing BACKGROUND_*.wav files to determine starting number
    existing_files = sorted(glob.glob(os.path.join(background_dir, "BACKGROUND_*.wav")))
    start_num = 0
    if existing_files:
        last_file = os.path.basename(existing_files[-1])
        last_num_str = last_file.replace("BACKGROUND_", "").replace(".wav", "")
        if last_num_str.isdigit():
            start_num = int(last_num_str)
    
    print(f"\nExisting BACKGROUND files: {len(existing_files)}")
    print(f"Starting new number from: {start_num + 1}")
    
    # Find all non-BACKGROUND .wav files
    all_files = glob.glob(os.path.join(background_dir, "*.wav"))
    new_files = [f for f in all_files if not os.path.basename(f).startswith("BACKGROUND_")]
    new_files = sorted(new_files)
    
    print(f"\nFound {len(new_files)} files to rename:")
    
    if not new_files:
        print("  No files to rename.")
        return
    
    # Rename files
    renamed_count = 0
    for idx, old_path in enumerate(new_files, start=start_num + 1):
        old_name = os.path.basename(old_path)
        new_name = f"BACKGROUND_{idx:03d}.wav"
        new_path = os.path.join(background_dir, new_name)
        
        try:
            os.rename(old_path, new_path)
            print(f"  ✓ {old_name:40s} → {new_name}")
            renamed_count += 1
        except Exception as e:
            print(f"  ✗ Failed to rename {old_name}: {e}")
    
    print(f"\n{'='*70}")
    print(f"Successfully renamed {renamed_count} files")
    print(f"New total: {len(existing_files) + renamed_count} BACKGROUND files")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
