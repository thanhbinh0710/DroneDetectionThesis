#!/usr/bin/env python3
"""
Script để đếm số lượng mẫu được dán nhãn
Kiểm tra metadata.csv và dữ liệu đã xử lý (processed data)
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def count_metadata_samples(metadata_path):
    """Đếm số lượng file gốc trong metadata.csv"""
    if not os.path.exists(metadata_path):
        print(f"❌ Không tìm thấy file metadata: {metadata_path}")
        return None
    
    df = pd.read_csv(metadata_path, sep=';', encoding='utf-8-sig')
    
    print("\n" + "="*70)
    print("📊 THỐNG KÊ DỮ LIỆU GỐC (metadata.csv)")
    print("="*70)
    
    total = len(df)
    drone_count = len(df[df['label'].str.upper() == 'DRONE'])
    not_drone_count = len(df[df['label'].str.upper() == 'NOT_DRONE'])
    
    print(f"\n✓ Tổng số file gốc: {total}")
    print(f"  • DRONE (máy bay không người lái): {drone_count} file")
    print(f"  • NOT_DRONE (không phải drone): {not_drone_count} file")
    print(f"  • Tỷ lệ: {drone_count}/{not_drone_count} (DRONE/NOT_DRONE)")
    
    return df

def count_processed_samples(processed_dir):
    """Đếm số lượng mẫu sau khi xử lý (đã phân đoạn và tăng cường)"""
    features_path = os.path.join(processed_dir, 'features.npy')
    labels_path = os.path.join(processed_dir, 'labels.npy')
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print("\n⚠️  Chưa có dữ liệu đã xử lý")
        print(f"   Không tìm thấy: {processed_dir}")
        print(f"   Hãy chạy: python -m src.training.data_loader")
        return None
    
    features = np.load(features_path)
    labels = np.load(labels_path)
    
    print("\n" + "="*70)
    print("📊 THỐNG KÊ DỮ LIỆU ĐÃ XỬ LÝ (features.npy & labels.npy)")
    print("="*70)
    
    total = len(labels)
    drone_count = np.sum(labels == 1)
    not_drone_count = np.sum(labels == 0)
    
    print(f"\n✓ Tổng số mẫu sau khi xử lý: {total}")
    print(f"  • DRONE (label=1): {drone_count} mẫu")
    print(f"  • NOT_DRONE (label=0): {not_drone_count} mẫu")
    print(f"  • Tỷ lệ: {drone_count}/{not_drone_count} (DRONE/NOT_DRONE)")
    print(f"\n📐 Kích thước dữ liệu:")
    print(f"  • Features shape: {features.shape}")
    print(f"  • Labels shape: {labels.shape}")
    
    return features, labels

def estimate_segmentation_info(metadata_df, processed_labels):
    """Ước tính thông tin về phân đoạn"""
    if metadata_df is None or processed_labels is None:
        return
    
    original_count = len(metadata_df)
    processed_count = len(processed_labels)
    
    if processed_count > original_count:
        factor = processed_count / original_count
        
        print("\n" + "="*70)
        print("🔍 PHÂN TÍCH PHÂN ĐOẠN & TĂNG CƯỜNG DỮ LIỆU")
        print("="*70)
        print(f"\n✓ Hệ số nhân: {factor:.2f}x")
        print(f"  • Ban đầu: {original_count} file")
        print(f"  • Sau xử lý: {processed_count} mẫu")
        print(f"  • Tăng thêm: {processed_count - original_count} mẫu")
        
        # Phân tích khả năng
        print(f"\n📌 Phân tích:")
        if factor >= 10:
            print(f"  • Có thể đã sử dụng cả phân đoạn (segmentation) VÀ tăng cường dữ liệu (augmentation)")
        elif factor >= 2:
            print(f"  • Có thể đã sử dụng phân đoạn (segmentation) hoặc tăng cường dữ liệu (augmentation)")
        else:
            print(f"  • Chưa sử dụng phân đoạn hoặc tăng cường dữ liệu mạnh")

def check_audio_files(raw_dir):
    """Kiểm tra file audio thực tế trong thư mục"""
    print("\n" + "="*70)
    print("📁 KIỂM TRA FILE AUDIO THỰC TẾ")
    print("="*70)
    
    drone_dir = os.path.join(raw_dir, 'drone')
    background_dir = os.path.join(raw_dir, 'background')
    
    drone_files = list(Path(drone_dir).glob('*.wav')) if os.path.exists(drone_dir) else []
    background_files = list(Path(background_dir).glob('*.wav')) if os.path.exists(background_dir) else []
    
    print(f"\n✓ File audio trong thư mục:")
    print(f"  • {drone_dir}")
    print(f"    → {len(drone_files)} file .wav")
    print(f"  • {background_dir}")
    print(f"    → {len(background_files)} file .wav")
    print(f"\n  Tổng: {len(drone_files) + len(background_files)} file .wav")

def main():
    """Hàm chính"""
    # Xác định đường dẫn
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    metadata_path = os.path.join(project_root, 'data', 'metadata.csv')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    raw_dir = os.path.join(project_root, 'data', 'raw')
    
    print("\n" + "="*70)
    print("🔢 CÔNG CỤ THỐNG KÊ MẪU DỮ LIỆU - DRONE DETECTION")
    print("="*70)
    print(f"\n📂 Project: {project_root}")
    
    # 1. Kiểm tra file thực tế
    check_audio_files(raw_dir)
    
    # 2. Đếm mẫu trong metadata
    metadata_df = count_metadata_samples(metadata_path)
    
    # 3. Đếm mẫu đã xử lý
    processed_data = count_processed_samples(processed_dir)
    
    # 4. Phân tích phân đoạn
    if processed_data is not None:
        features, labels = processed_data
        estimate_segmentation_info(metadata_df, labels)
    
    # Tổng kết
    print("\n" + "="*70)
    print("💡 HƯỚNG DẪN")
    print("="*70)
    print("""
Để xử lý lại dữ liệu với phân đoạn (segmentation), hãy chỉnh sửa:
  
  📝 File: src/training/data_loader.py
  
  Tìm đoạn code (ở cuối file) và thay đổi:
  
    # Load dataset
    X, y, filenames = load_audio_dataset(
        data_dir=data_dir,
        metadata_path=metadata_path,
        augment=True,                     # Có tăng cường dữ liệu
        augment_factor=3,                 # Tạo 3 phiên bản augmented/mẫu
        target_length=128,                # Độ dài time-step
        use_segmentation=True,            # ← BẬT PHÂN ĐOẠN
        segment_duration=1.0,             # Độ dài mỗi đoạn (giây)
        segment_overlap=0.5               # Chồng lấp 50%
    )
  
  Sau đó chạy lại:
    python -m src.training.data_loader
    
  Để kiểm tra lại kết quả:
    python scripts/count_samples.py
""")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
