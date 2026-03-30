#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to automatically detect background audio files and update metadata.csv
Usage: python scripts/add_background_samples.py
"""

import os
import sys
import glob
import pandas as pd
from pathlib import Path

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)


def main():
    """Scan background folder and update metadata.csv"""
    
    # Paths
    background_dir = os.path.join(project_root, "data", "raw", "background")
    metadata_path = os.path.join(project_root, "data", "metadata.csv")
    
    print("="*70)
    print("BACKGROUND AUDIO SAMPLES - METADATA UPDATER")
    print("="*70)
    
    # Check if background folder exists
    if not os.path.exists(background_dir):
        print(f"\n❌ Error: Background folder not found: {background_dir}")
        print("   Please create the folder first and add audio files.")
        return
    
    # Find all .wav files in background folder
    background_files = sorted(glob.glob(os.path.join(background_dir, "*.wav")))
    
    if not background_files:
        print(f"\n⚠️  Warning: No .wav files found in {background_dir}")
        print("   Please add background audio files (BACKGROUND_001.wav, BACKGROUND_002.wav, ...)")
        return
    
    print(f"\n✓ Found {len(background_files)} background audio files:")
    for idx, fp in enumerate(background_files, 1):
        size_mb = os.path.getsize(fp) / (1024 * 1024)
        print(f"  {idx:2d}. {os.path.basename(fp):30s} ({size_mb:.2f} MB)")
    
    # Load existing metadata
    if not os.path.exists(metadata_path):
        print(f"\n❌ Error: Metadata file not found: {metadata_path}")
        return
    
    df = pd.read_csv(metadata_path, sep=';', encoding='utf-8-sig')
    print(f"\n✓ Current metadata: {len(df)} entries")
    print(f"  - DRONE samples: {len(df[df['label'] == 'DRONE'])}")
    
    # Check for existing background entries
    existing_background = df[df['label'] == 'NOT_DRONE']
    print(f"  - NOT_DRONE samples: {len(existing_background)}")
    
    # Prepare new entries
    new_entries = []
    for fp in background_files:
        filename_rel = os.path.join("background", os.path.basename(fp))
        
        # Check if already in metadata
        if filename_rel in df['filename'].values:
            print(f"  ⚠️  Skipping (already exists): {filename_rel}")
            continue
        
        new_entry = {
            'filename': filename_rel,
            'label': 'NOT_DRONE',
            'class_id': 0,
            'source': 'background',
            'notes': f'Background noise sample'
        }
        new_entries.append(new_entry)
    
    if not new_entries:
        print("\n✓ No new entries to add. All background files already in metadata.")
        return
    
    # Add new entries to dataframe
    df_new = pd.DataFrame(new_entries)
    df_updated = pd.concat([df, df_new], ignore_index=True)
    
    print(f"\n📝 Adding {len(new_entries)} new NOT_DRONE entries to metadata...")
    
    # Backup original metadata
    backup_path = metadata_path + ".backup"
    df.to_csv(backup_path, sep=';', index=False, encoding='utf-8-sig')
    print(f"✓ Backup saved: {os.path.basename(backup_path)}")
    
    # Save updated metadata
    df_updated.to_csv(metadata_path, sep=';', index=False, encoding='utf-8-sig')
    print(f"✓ Updated metadata saved: {os.path.basename(metadata_path)}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total samples: {len(df_updated)}")
    print(f"  - DRONE:     {len(df_updated[df_updated['label'] == 'DRONE'])} samples")
    print(f"  - NOT_DRONE: {len(df_updated[df_updated['label'] == 'NOT_DRONE'])} samples")
    
    drone_count = len(df_updated[df_updated['label'] == 'DRONE'])
    not_drone_count = len(df_updated[df_updated['label'] == 'NOT_DRONE'])
    
    if drone_count != not_drone_count:
        print(f"\n⚠️  WARNING: Imbalanced dataset!")
        print(f"   For best results, try to have equal numbers of DRONE and NOT_DRONE samples.")
        if drone_count > not_drone_count:
            print(f"   Add {drone_count - not_drone_count} more NOT_DRONE samples.")
        else:
            print(f"   You have {not_drone_count - drone_count} more NOT_DRONE than DRONE samples.")
    else:
        print(f"\n✓ Dataset is balanced! ({drone_count} samples per class)")
    
    print("\n✓ Done! You can now run: python -m src.training.data_loader")
    print("="*70)


if __name__ == "__main__":
    main()
