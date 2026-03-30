"""
Script demo: Kiểm tra tính năng cắt đoạn audio (Audio Segmentation)
Hiển thị cách audio dài được cắt thành các đoạn nhỏ với sliding window
"""

import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.data_loader import segment_audio
from src.common.processor import preprocess_audio


def visualize_segmentation(audio_path, segment_duration=1.0, overlap=0.5, sr=44100):
    """
    Hiển thị cách audio được cắt thành các đoạn nhỏ
    
    Args:
        audio_path: Đường dẫn file audio
        segment_duration: Độ dài mỗi đoạn (giây)
        overlap: Tỷ lệ chồng lấp (0.0-1.0)
        sr: Sample rate
    """
    print("="*70)
    print("AUDIO SEGMENTATION VISUALIZATION")
    print("="*70)
    
    # Load audio
    print(f"\n📂 Loading: {os.path.basename(audio_path)}")
    audio = preprocess_audio(audio_path, sr=sr)
    
    duration = len(audio) / sr
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Samples: {len(audio)}")
    
    # Segment audio
    print(f"\n✂️  Segmenting...")
    print(f"   Segment duration: {segment_duration}s")
    print(f"   Overlap: {overlap*100}%")
    
    segments = segment_audio(audio, sr=sr, 
                            segment_duration=segment_duration, 
                            overlap=overlap)
    
    print(f"\n✓ Created {len(segments)} segments")
    
    # Calculate time positions
    segment_samples = int(segment_duration * sr)
    hop_samples = int(segment_samples * (1 - overlap))
    
    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Original audio with segment markers
    time = np.arange(len(audio)) / sr
    axes[0].plot(time, audio, linewidth=0.5, alpha=0.7)
    axes[0].set_title(f'Original Audio ({duration:.2f}s)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Add segment boundaries
    for i in range(len(segments)):
        start_time = i * hop_samples / sr
        end_time = start_time + segment_duration
        axes[0].axvline(x=start_time, color='red', linestyle='--', alpha=0.5)
        axes[0].axvline(x=end_time, color='blue', linestyle='--', alpha=0.5)
        axes[0].axvspan(start_time, end_time, alpha=0.1, color='green')
    
    # Plot 2: All segments overlaid
    for i, segment in enumerate(segments):
        seg_time = np.arange(len(segment)) / sr
        axes[1].plot(seg_time, segment, linewidth=0.5, alpha=0.5, 
                    label=f'Segment {i+1}')
    
    axes[1].set_title(f'{len(segments)} Segments ({segment_duration}s each, {overlap*100}% overlap)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', ncol=3, fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print segment details
    print("\n" + "="*70)
    print("SEGMENT DETAILS")
    print("="*70)
    for i in range(len(segments)):
        start_time = i * hop_samples / sr
        end_time = start_time + segment_duration
        print(f"Segment {i+1:2d}: {start_time:6.2f}s - {end_time:6.2f}s  "
              f"({len(segments[i])} samples)")
    print("="*70)


if __name__ == "__main__":
    # Tìm file audio trong thư mục data/raw/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    # Thử tìm file DRONE đầu tiên
    drone_dir = os.path.join(project_root, "data", "raw", "drone")
    background_dir = os.path.join(project_root, "data", "raw", "background")
    
    audio_file = None
    
    # Tìm file trong thư mục drone
    if os.path.exists(drone_dir):
        drone_files = [f for f in os.listdir(drone_dir) if f.endswith('.wav')]
        if drone_files:
            audio_file = os.path.join(drone_dir, drone_files[0])
    
    # Nếu không có, thử tìm trong background
    if audio_file is None and os.path.exists(background_dir):
        bg_files = [f for f in os.listdir(background_dir) if f.endswith('.wav')]
        if bg_files:
            audio_file = os.path.join(background_dir, bg_files[0])
    
    if audio_file is None:
        print("❌ Error: No .wav files found in data/raw/drone/ or data/raw/background/")
        print("   Please add some audio files to test segmentation.")
        exit(1)
    
    # Demo với các cấu hình khác nhau
    print("\n🎵 Testing audio segmentation with different configurations...\n")
    
    # Config 1: 1s segments, 50% overlap (RECOMMENDED)
    print("\n" + "█"*70)
    print("   CONFIG 1: 1s segments, 50% overlap (RECOMMENDED)")
    print("█"*70)
    visualize_segmentation(audio_file, segment_duration=1.0, overlap=0.5)
    
    # Uncomment để test các config khác:
    # Config 2: 2s segments, 25% overlap
    # print("\n" + "█"*70)
    # print("   CONFIG 2: 2s segments, 25% overlap")  
    # print("█"*70)
    # visualize_segmentation(audio_file, segment_duration=2.0, overlap=0.25)
    
    # Config 3: 0.5s segments, 75% overlap (nhiều segments)
    # print("\n" + "█"*70)
    # print("   CONFIG 3: 0.5s segments, 75% overlap")
    # print("█"*70)
    # visualize_segmentation(audio_file, segment_duration=0.5, overlap=0.75)
