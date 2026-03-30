import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def preprocess_audio(file_path, sr=44100):
    # 1. Tải và chuẩn hóa dữ liệu (Normalization)
    # Tự động resample về cùng một tỷ lệ lấy mẫu (Sample Rate)
    y, _ = librosa.load(file_path, sr=sr)
    
    # Chuẩn hóa biên độ về khoảng [-1, 1]
    y = librosa.util.normalize(y)
    
    # 2. Loại bỏ khoảng lặng (Silence Removal)
    # Loại bỏ các đoạn âm thanh có năng lượng thấp dưới 20dB
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    
    return y_trimmed

def extract_mel_spectrogram(y, sr=44100):
    # 3. Phân đoạn và tính Mel-Spectrogram (Framing & Windowing) [cite: 12, 13, 14]
    # n_fft: Độ dài khung (Window length)
    # hop_length: Độ chồng lấp (Overlap) [cite: 15]
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, 
        n_fft=2048, 
        hop_length=512, 
        n_mels=128 # Số lượng bộ lọc Mel [cite: 40, 41]
    )
    
    # 4. Chuyển đổi sang thang đo Log-Power (dB) [cite: 45, 46]
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db

# --- CÁC KỸ THUẬT TĂNG CƯỜNG DỮ LIỆU (AUGMENTATION) ---

def add_noise(y, noise_factor=0.0005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def pitch_shift(y, sr, steps=6):
    # Mô phỏng sự thay đổi tần số động cơ drone
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def time_shift(y, shift_max=0.4):
    # Dịch chuyển thời gian ngẫu nhiên
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)

# --- CHƯƠNG TRÌNH CHÍNH MINH HỌA ---
if __name__ == "__main__":
    import os
    import glob
    
    # Xác định đường dẫn data/raw/drone (tương đối từ thư mục gốc project)
    # Tìm thư mục gốc project (chứa folder data/)
    script_dir = os.path.dirname(os.path.abspath(__file__))  # src/common/
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))  # Lên 2 cấp
    drone_dir = os.path.join(project_root, "data", "raw", "drone")
    
    print(f"📂 Đường dẫn drone data: {drone_dir}\n")
    
    # Tìm tất cả file DRONE trong thư mục data/raw/drone
    drone_files = sorted(glob.glob(os.path.join(drone_dir, "DRONE_*.wav")))
    
    if not drone_files:
        print(f"❌ Lỗi: Không tìm thấy file DRONE trong thư mục '{drone_dir}'")
        print(f"   Vui lòng đặt file audio với tên dạng 'DRONE_001.wav', 'DRONE_002.wav', ...")
        print(f"   vào thư mục: data/raw/drone/")
        exit(1)
    
    # Hiển thị danh sách file tìm được
    print(f"✓ Tìm thấy {len(drone_files)} file DRONE:")
    for idx, file in enumerate(drone_files, 1):
        print(f"  {idx}. {os.path.basename(file)}")
    
    # Lấy file đầu tiên để xử lý
    audio_path = drone_files[0]
    print(f"\n→ Đang xử lý: {os.path.basename(audio_path)}\n")
    
    # Bước tiền xử lý cơ bản
    y_clean = preprocess_audio(audio_path)
    
    # Trích xuất đặc trưng gốc
    mel_original = extract_mel_spectrogram(y_clean)
    
    # === 3 PHIÊN BẢN TĂNG CƯỜNG DỮ LIỆU KHÁC NHAU ===
    
    # Phiên bản 1: Augmentation NHẸ (Light) - Thay đổi tinh tế
    noise_factor_1 = 0.0002
    steps_1 = 2
    shift_max_1 = 0.2
    y_aug1 = add_noise(y_clean, noise_factor=noise_factor_1)
    y_aug1 = pitch_shift(y_aug1, sr=44100, steps=steps_1)
    y_aug1 = time_shift(y_aug1, shift_max=shift_max_1)
    mel_aug1 = extract_mel_spectrogram(y_aug1)
    
    # Phiên bản 2: Augmentation VỪA (Medium) - Thay đổi trung bình
    noise_factor_2 = 0.0005
    steps_2 = 4
    shift_max_2 = 0.4
    y_aug2 = add_noise(y_clean, noise_factor=noise_factor_2)
    y_aug2 = pitch_shift(y_aug2, sr=44100, steps=steps_2)
    y_aug2 = time_shift(y_aug2, shift_max=shift_max_2)
    mel_aug2 = extract_mel_spectrogram(y_aug2)
    
    # Phiên bản 3: Augmentation MẠNH (Heavy) - Thay đổi rõ rệt
    noise_factor_3 = 0.001
    steps_3 = 7
    shift_max_3 = 0.6
    y_aug3 = add_noise(y_clean, noise_factor=noise_factor_3)
    y_aug3 = pitch_shift(y_aug3, sr=44100, steps=steps_3)
    y_aug3 = time_shift(y_aug3, shift_max=shift_max_3)
    mel_aug3 = extract_mel_spectrogram(y_aug3)
    
    # Hiển thị để kiểm tra (Dùng cho báo cáo mục III/IV)
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    librosa.display.specshow(mel_original, x_axis='time', y_axis='mel', sr=44100)
    plt.title(f'Original Mel-Spectrogram ({os.path.basename(audio_path)})')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(2, 2, 2)
    librosa.display.specshow(mel_aug1, x_axis='time', y_axis='mel', sr=44100)
    plt.title(f'(noise={noise_factor_1}, steps={steps_1}, shift={shift_max_1})')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(2, 2, 3)
    librosa.display.specshow(mel_aug2, x_axis='time', y_axis='mel', sr=44100)
    plt.title(f'(noise={noise_factor_2}, steps={steps_2}, shift={shift_max_2})')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(2, 2, 4)
    librosa.display.specshow(mel_aug3, x_axis='time', y_axis='mel', sr=44100)
    plt.title(f'(noise={noise_factor_3}, steps={steps_3}, shift={shift_max_3})')
    plt.colorbar(format='%+2.0f dB')
    
    plt.tight_layout()
    plt.show()
    
    # In ra thông số để so sánh
    print("\n=== SO SÁNH CÁC PHIÊN BẢN TĂNG CƯỜNG DỮ LIỆU ===")
    print(f"Phiên bản 1 (NHẸ):  noise_factor={noise_factor_1}, steps={steps_1}, shift_max={shift_max_1}")
    print(f"Phiên bản 2 (VỪA):  noise_factor={noise_factor_2}, steps={steps_2}, shift_max={shift_max_2}")
    print(f"Phiên bản 3 (MẠNH): noise_factor={noise_factor_3}, steps={steps_3}, shift_max={shift_max_3}")
    
    # Lưu dưới dạng mảng NumPy phục vụ huấn luyện (Mục IV)
    # np.save("drone_feature.npy", mel_original)
