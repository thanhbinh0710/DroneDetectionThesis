# src/training/data_loader.py
"""
Audio Data Loader Module
Load and prepare audio datasets for drone detection training
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.common.processor import preprocess_audio, extract_mel_spectrogram, add_noise, pitch_shift, time_shift


def segment_audio(audio, sr=16000, segment_duration=1.0, overlap=0.5):
    """
    Cắt audio thành các đoạn nhỏ với sliding window
    
    Args:
        audio: Audio waveform array
        sr: Sample rate (Hz)
        segment_duration: Độ dài mỗi đoạn (giây)
        overlap: Tỷ lệ chồng lấp giữa các đoạn (0.0-1.0)
                 0.5 nghĩa là mỗi đoạn chồng lấp 50% với đoạn trước
        
    Returns:
        List of audio segments
    """
    segment_samples = int(segment_duration * sr)  # Số samples trong 1 đoạn
    hop_samples = int(segment_samples * (1 - overlap))  # Bước nhảy giữa các đoạn
    
    segments = []
    start = 0
    
    while start + segment_samples <= len(audio):
        segment = audio[start:start + segment_samples]
        segments.append(segment)
        start += hop_samples
    
    # Xử lý đoạn cuối (nếu còn dư)
    if start < len(audio) and len(audio) - start > segment_samples * 0.5:
        # Chỉ thêm nếu đoạn cuối dài hơn 50% độ dài segment
        segment = audio[-segment_samples:]  # Lấy từ cuối
        segments.append(segment)

    # Đảm bảo luôn có ít nhất một segment: pad ngắn thành đúng độ dài segment
    if len(segments) == 0:
        if len(audio) >= segment_samples:
            segments.append(audio[:segment_samples])
        else:
            pad_width = segment_samples - len(audio)
            seg = np.pad(audio, (0, pad_width), mode='constant', constant_values=0)
            segments.append(seg)
    
    return segments


def pad_or_truncate_spectrogram(mel_spec, target_length=128):
    """
    Pad or truncate mel-spectrogram to fixed length
    
    Args:
        mel_spec: Mel-spectrogram array (n_mels, time_steps)
        target_length: Target number of time steps
        
    Returns:
        Padded or truncated mel-spectrogram (n_mels, target_length)
    """
    current_length = mel_spec.shape[1]
    
    if current_length > target_length:
        # Truncate: take first target_length frames
        return mel_spec[:, :target_length]
    elif current_length < target_length:
        # Pad: add zeros to the end
        pad_width = target_length - current_length
        return np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    else:
        return mel_spec


def load_metadata(metadata_path):
    """
    Load metadata CSV file containing audio file labels
    
    Args:
        metadata_path: Path to metadata.csv
        
    Returns:
        DataFrame with columns: filename, label, source, duration_sec, notes
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    df = pd.read_csv(metadata_path, sep=';', encoding='utf-8-sig')
    print(f"Loaded metadata: {len(df)} files")
    return df


def load_audio_dataset(data_dir, metadata_path, augment=False, augment_factor=3, target_length=128,
                       use_segmentation=False, segment_duration=1.0, segment_overlap=0.5):
    """
    Load audio files and extract mel-spectrogram features for drone detection
    
    Args:
        data_dir: Directory containing audio files (.wav)
        metadata_path: Path to metadata CSV
        augment: Whether to apply data augmentation
        augment_factor: Number of augmented versions per original sample
        target_length: Target time steps for mel-spectrogram (default: 128)
        use_segmentation: Có cắt audio thành các đoạn nhỏ hay không
        segment_duration: Độ dài mỗi đoạn (giây) khi sử dụng segmentation
        segment_overlap: Tỷ lệ chồng lấp giữa các đoạn (0.0-1.0)
        
    Returns:
        X: Feature array - mel-spectrograms, shape (n_samples, 128, target_length)
        y: Label array - 1 for DRONE, 0 for NOT_DRONE
        filenames: List of processed filenames
        source_file_indices: Array mapping each sample to its source file index in metadata
    """
    metadata = load_metadata(metadata_path)
    
    features = []
    labels = []
    filenames = []
    source_file_indices = []  # Track which original file each sample came from
    
    print(f"\n{'='*60}")
    print("PROCESSING AUDIO FILES")
    print(f"{'='*60}")
    print(f"Target shape: (128 mel-bands, {target_length} time-steps)")
    if use_segmentation:
        print(f"Segmentation: ENABLED ({segment_duration}s segments, {segment_overlap*100}% overlap)")
    else:
        print(f"Segmentation: DISABLED (whole file processing)")
    print()
    
    for idx, row in metadata.iterrows():
        filepath = os.path.join(data_dir, row['filename'])
        
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue
        
        print(f"Processing [{idx+1}/{len(metadata)}]: {row['filename']}")
        
        # Preprocess audio
        audio = preprocess_audio(filepath)
        
        # Convert label to binary (1=DRONE, 0=NOT_DRONE)
        label = 1 if row['label'].upper() == 'DRONE' else 0

        # Mảng lưu trữ các phiên bản audio nguyên tác và augmentation
        audio_versions = [(audio, "")]

        # Thực hiện data augmentation trên âm thanh gốc TRƯỚC
        if augment and label == 1:
            for aug_idx in range(augment_factor):
                # Apply random augmentation on the full audio
                audio_aug = add_noise(audio, noise_factor=np.random.uniform(0.0002, 0.001))
                audio_aug = pitch_shift(audio_aug, sr=16000, steps=np.random.randint(-4, 5))
                audio_aug = time_shift(audio_aug, shift_max=0.4)
                audio_versions.append((audio_aug, f"_aug{aug_idx+1}"))
            print(f"  -> Generated {augment_factor} augmented full-audio versions for DRONE class")

        segment_count = 0
        for current_audio, aug_suffix in audio_versions:
            # Decide whether to segment or process whole audio
            if use_segmentation:
                # Cắt audio thành các đoạn nhỏ
                audio_segments = segment_audio(current_audio, sr=16000, 
                                              segment_duration=segment_duration, 
                                              overlap=segment_overlap)
            else:
                # Xử lý toàn bộ file
                audio_segments = [current_audio]
            
            segment_count += len(audio_segments)

            # Process each segment
            for seg_idx, audio_seg in enumerate(audio_segments):
                # Extract mel-spectrogram features
                mel_spec = extract_mel_spectrogram(audio_seg)
                
                # Pad or truncate to fixed length
                mel_spec = pad_or_truncate_spectrogram(mel_spec, target_length)
                
                # Add segment sample
                features.append(mel_spec)
                labels.append(label)
                source_file_indices.append(idx)  # Track source file index
                
                seg_suffix = f"_seg{seg_idx+1}" if (use_segmentation and len(audio_segments) > 1) else ""
                filenames.append(f"{row['filename']}{seg_suffix}{aug_suffix}")
                
        if use_segmentation:
            print(f"  -> Segmented into {segment_count} segments total ({segment_duration}s each)")
    
    print(f"\nTotal samples loaded: {len(features)}")
    print(f"  - Original files: {len(metadata)}")
    if use_segmentation:
        print(f"  - After segmentation: {len([f for f in filenames if '_seg' in f and '_aug' not in f])}")
    if augment:
        print(f"  - After augmentation: {len(features)}")
    
    # Convert lists to numpy arrays with explicit dtypes
    features_np = np.array(features, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.int64)
    source_file_indices_np = np.array(source_file_indices, dtype=np.int32)

    return features_np, labels_np, filenames, source_file_indices_np


def save_processed_features(features, labels, output_dir, source_file_indices=None):
    """
    Save processed mel-spectrogram features to disk
    
    Args:
        features: Feature array (mel-spectrograms)
        labels: Label array (1=DRONE, 0=NOT_DRONE)
        output_dir: Output directory (e.g., data/processed/)
        source_file_indices: Array mapping each sample to its source file index
    """
    os.makedirs(output_dir, exist_ok=True)
    
    feature_path = os.path.join(output_dir, 'features.npy')
    label_path = os.path.join(output_dir, 'labels.npy')
    
    # Ensure numpy arrays and expected dtype
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)

    # Desired feature shape: (N, 128, 128)
    if features.ndim == 3:
        # If axes are swapped (N, 128, 20), transpose to (N, 20, 128)
        if features.shape[1] == 128 and features.shape[2] == 20:
            features = features.transpose(0, 2, 1)

    np.save(feature_path, features)
    np.save(label_path, labels)
    
    print(f"\nSaved {len(features)} samples to {output_dir}")
    print(f"  - Features: {feature_path} (shape: {features.shape})")
    print(f"  - Labels: {label_path} (shape: {labels.shape})")
    
    # Save source file indices if provided
    if source_file_indices is not None:
        source_path = os.path.join(output_dir, 'source_file_indices.npy')
        np.save(source_path, source_file_indices)
        print(f"  - Source file indices: {source_path} (shape: {source_file_indices.shape})")


if __name__ == "__main__":
    # Set seeds for reproducibility of augmentation random ops
    random.seed(42)
    np.random.seed(42)
    # Example usage: Load and process audio dataset
    # Xác định đường dẫn tuyệt đối
    script_dir = os.path.dirname(os.path.abspath(__file__))  # src/training/
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))  # Lên 2 cấp
    data_dir = os.path.join(project_root, "data", "raw")
    metadata_path = os.path.join(project_root, "data", "metadata.csv")
    output_dir = os.path.join(project_root, "data", "processed")
    
    print("=" * 60)
    print("AUDIO DATA LOADER - DRONE DETECTION")
    print("=" * 60)
    print(f"\nProject root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Metadata file: {metadata_path}\n")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"\nError: Data directory not found: {data_dir}")
        print("   Please create the directory and add audio files (.wav)")
        exit(1)
    
    if not os.path.exists(metadata_path):
        print(f"\nError: Metadata file not found: {metadata_path}")
        print("   Please create metadata.csv with columns: filename, label, source, duration_sec, notes")
        exit(1)
    
    # ============================================================
    # CHỌN CHẾ ĐỘ XỬ LÝ DỮ LIỆU
    # ============================================================
    # Mode 1: Không segmentation, không augmentation
    # X, y, filenames = load_audio_dataset(data_dir, metadata_path)
    
    # Mode 2: Có segmentation, không augmentation (KHUYẾN NGHỊ cho file audio dài)
    # X, y, filenames = load_audio_dataset(data_dir, metadata_path, 
    #                                       use_segmentation=True, 
    #                                       segment_duration=1.0,
    #                                       segment_overlap=0.5)
    
    # Mode 3: Có augmentation, không segmentation
    # X, y, filenames = load_audio_dataset(data_dir, metadata_path, 
    #                                       augment=True, 
    #                                       augment_factor=3)
    
    # Mode 4: CẢ SEGMENTATION VÀ AUGMENTATION (Tăng dữ liệu tối đa)
    print("\nLoading dataset with SEGMENTATION and AUGMENTATION...")
    print("Configuration:")
    print("   - Segmentation: 1.0s segments with 50% overlap")
    print("   - Augmentation: ENABLED (factor=3)\n")
    
    try:
        X, y, filenames, source_file_indices = load_audio_dataset(
            data_dir, metadata_path, 
            augment=True,
            augment_factor=3,
            use_segmentation=True,
            segment_duration=1.0,
            segment_overlap=0.5
        )
        
        if len(X) > 0:
            print(f"\n{'='*60}")
            print("DATASET SUMMARY")
            print(f"{'='*60}")
            print(f"Total samples: {len(X)}")
            print(f"Feature shape: {X[0].shape} (mel_bands × time_steps)")
            print(f"Label distribution:")
            print(f"  - DRONE: {np.sum(y == 1)}")
            print(f"  - NOT_DRONE: {np.sum(y == 0)}")
            
            # Save processed features with source file indices
            save_processed_features(X, y, output_dir, source_file_indices)
        else:
            print("\nWarning: No audio files processed. Please check your data directory.")
    
    except Exception as e:
        print(f"\nError processing dataset: {str(e)}")
        import traceback
        traceback.print_exc()
