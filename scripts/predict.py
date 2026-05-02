"""
Inference Script with Majority Voting for Drone Audio Detection
Usage: python scripts/predict.py --audio data/raw/ten_file_test.wav
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Đảm bảo import được các module trong dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.data_loader import segment_audio, pad_or_truncate_spectrogram
from src.common.processor import preprocess_audio, extract_mel_spectrogram

def predict_file(audio_path, model_path):
    print(f"{'='*50}")
    print(f"ĐANG PHÂN TÍCH AUDIO: {os.path.basename(audio_path)}")
    print(f"{'='*50}")

    if not os.path.exists(audio_path):
        print("Lỗi: Không tìm thấy file audio!")
        return

    # 1. Load mô hình CNN đã train
    model = tf.keras.models.load_model(model_path)

    # 2. Tiền xử lý và cắt file thành các đoạn 1 giây (overlap 50%)
    audio = preprocess_audio(audio_path)
    segments = segment_audio(audio, sr=16000, segment_duration=1.0, overlap=0.5)

    if len(segments) == 0:
        print("File audio quá ngắn, không thể xử lý.")
        return

    # 3. Trích xuất đặc trưng Mel-Spectrogram cho từng đoạn
    X = []
    for seg in segments:
        spec = extract_mel_spectrogram(seg)
        spec = pad_or_truncate_spectrogram(spec, target_length=128)
        X.append(spec)

    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1) # Định dạng lại cho CNN

    # 4. CNN Dự đoán từng giây
    y_pred_proba = model.predict(X, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.4).astype(int)

    print(f"Tổng số phân đoạn cắt được: {len(segments)}")
    print(f"Kết quả dự đoán gốc (Từng giây): {y_pred}\n")

    # ==========================================
    # 5. THUẬT TOÁN MAJORITY VOTING (2/3)
    # ==========================================
    window_size = 3
    threshold = 2
    final_decision = 0
    alarm_zones = []

    if len(y_pred) < window_size:
        # Fallback: Dành cho file cực ngắn (dưới 1.5 giây)
        print("[!] File quá ngắn. Kích hoạt chế độ Fallback: Bất kỳ đoạn nào báo 1 -> Có Drone.")
        final_decision = 1 if sum(y_pred) > 0 else 0
    else:
        # Trượt cửa sổ 3 đoạn liên tiếp
        for i in range(len(y_pred) - window_size + 1):
            window = y_pred[i : i + window_size]
            if sum(window) >= threshold:
                final_decision = 1
                # Tính toán thời gian vật lý (Mỗi bước nhảy hop_length là 0.5s do overlap 50%)
                start_time = i * 0.5
                end_time = (i + window_size) * 0.5
                alarm_zones.append(f"[{start_time:.1f}s - {end_time:.1f}s]")

    # 6. Đưa ra phán quyết cuối cùng
    if final_decision == 1:
        print("🚨 KẾT LUẬN: CÓ DRONE TRONG KHU VỰC! 🚨")
        print("Hệ thống chốt cảnh báo nhờ Majority Voting tại các khung thời gian:")
        # Chỉ in ra vài khoảng thời gian đầu để tránh spam màn hình
        print(", ".join(alarm_zones[:5]) + ("..." if len(alarm_zones) > 5 else ""))
    else:
        print("✅ KẾT LUẬN: MÔI TRƯỜNG AN TOÀN (Không phát hiện Drone). ✅")
        if sum(y_pred) > 0:
            print(f"(Hệ thống đã tự động lọc bỏ {sum(y_pred)} cảnh báo nhiễu 1 giây nhờ luật 2/3)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, required=True, help='Đường dẫn tới file .wav cần test')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    model_path = project_root / 'models' / 'drone_model_v1.keras' # Đổi tên nếu mô hình bạn tên khác

    predict_file(args.audio, model_path)