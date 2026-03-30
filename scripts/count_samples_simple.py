#!/usr/bin/env python3
"""
Script đơn giản để đếm số lượng mẫu - không cần thư viện ngoài
"""

import os
import sys

def count_metadata():
    """Đếm mẫu trong metadata.csv"""
    metadata_path = 'data/metadata.csv'
    
    if not os.path.exists(metadata_path):
        print(f"❌ Không tìm thấy {metadata_path}")
        return
    
    with open(metadata_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    # Bỏ dòng header
    data_lines = lines[1:]
    
    drone_count = 0
    not_drone_count = 0
    
    for line in data_lines:
        if not line.strip():
            continue
        parts = line.split(';')
        if len(parts) >= 2:
            label = parts[1].strip().upper()
            if label == 'DRONE':
                drone_count += 1
            elif label == 'NOT_DRONE':
                not_drone_count += 1
    
    total = drone_count + not_drone_count
    
    print("="*70)
    print("📊 THỐNG KÊ DỮ LIỆU GỐC (metadata.csv)")
    print("="*70)
    print(f"\n✓ Tổng số file gốc: {total}")
    print(f"  • DRONE: {drone_count} file")
    print(f"  • NOT_DRONE: {not_drone_count} file")
    print(f"  • Tỷ lệ: {drone_count}:{not_drone_count}")
    
    return total

def count_processed():
    """Đếm mẫu đã xử lý"""
    try:
        import numpy as np
        
        features_path = 'data/processed/features.npy'
        labels_path = 'data/processed/labels.npy'
        
        if not os.path.exists(features_path):
            print("\n⚠️  Chưa có dữ liệu đã xử lý")
            print("   Hãy chạy: python -m src.training.data_loader")
            return
        
        features = np.load(features_path)
        labels = np.load(labels_path)
        
        total = len(labels)
        drone_count = int(np.sum(labels == 1))
        not_drone_count = int(np.sum(labels == 0))
        
        print("\n" + "="*70)
        print("📊 THỐNG KÊ DỮ LIỆU ĐÃ XỬ LÝ (sau phân đoạn & tăng cường)")
        print("="*70)
        print(f"\n✓ Tổng số mẫu: {total}")
        print(f"  • DRONE (label=1): {drone_count} mẫu")
        print(f"  • NOT_DRONE (label=0): {not_drone_count} mẫu")
        print(f"  • Tỷ lệ: {drone_count}:{not_drone_count}")
        print(f"\n📐 Kích thước:")
        print(f"  • Features: {features.shape}")
        print(f"  • Labels: {labels.shape}")
        
        # Tính hệ số nhân
        original = count_metadata()
        if original and original > 0:
            factor = total / original
            print(f"\n🔍 Hệ số nhân dữ liệu: {factor:.1f}x")
            print(f"  • {total} mẫu từ {original} file gốc")
        
    except ImportError:
        print("\n⚠️  Cần cài đặt numpy để đọc dữ liệu đã xử lý")
        print("   Chạy: pip install numpy")

def count_audio_files():
    """Đếm file .wav thực tế"""
    import glob
    
    drone_files = glob.glob('data/raw/drone/*.wav')
    background_files = glob.glob('data/raw/background/*.wav')
    
    print("\n" + "="*70)
    print("📁 FILE AUDIO THỰC TẾ")
    print("="*70)
    print(f"\n✓ Số file .wav:")
    print(f"  • data/raw/drone/: {len(drone_files)} file")
    print(f"  • data/raw/background/: {len(background_files)} file")
    print(f"  • Tổng: {len(drone_files) + len(background_files)} file")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔢 THỐNG KÊ MẪU DỮ LIỆU - DRONE DETECTION PROJECT")
    print("="*70 + "\n")
    
    # 1. Đếm file thực tế
    count_audio_files()
    
    # 2. Đếm trong metadata
    print()
    original_count = count_metadata()
    
    # 3. Đếm mẫu đã xử lý
    count_processed()
    
    print("\n" + "="*70)
    print("💡 LƯU Ý")
    print("="*70)
    print("""
Khi BẬT phân đoạn dữ liệu (segmentation):
  • Mỗi file audio sẽ được cắt thành nhiều đoạn nhỏ
  • Ví dụ: 1 file 10 giây → 10 đoạn 1 giây (với overlap 50%)
  • Kết hợp tăng cường dữ liệu (augmentation): số mẫu tăng lên rất nhiều

Để bật segmentation, chỉnh sửa src/training/data_loader.py:
  use_segmentation=True
  segment_duration=1.0        # Độ dài mỗi đoạn (giây)
  segment_overlap=0.5         # Chồng lấp 50%
""")
    print("="*70 + "\n")
