import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def visualize_mel_filterbanks():
    """
    Tạo và trực quan hóa các bộ lọc Mel (Mel Filterbanks) dựa trên thông số
    hiện tại của dự án: sr = 16000, n_fft = 2048, n_mels = 128
    """
    # Thông số chuẩn của dự án
    sr = 16000
    n_fft = 2048
    n_mels = 128
    
    print(f"Đang tạo Mel Filterbanks với sr={sr}Hz, n_fft={n_fft}, n_mels={n_mels}...")
    
    # Lấy ma trận trọng số của Mel filters từ librosa
    mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    
    # Tạo thư mục assets để lưu file ảnh
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets'))
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Hình 1: Vẽ toàn bộ ma trận Mel Filterbank (Spectrogram-style) ---
    plt.figure(figsize=(10, 6))
    # Trục x là tần số tuyến tính (hiển thị theo Hz lên đến sr/2)
    librosa.display.specshow(mel_filters, x_axis='linear', sr=sr)
    plt.colorbar(format='%+2.0f')
    plt.title(f'Mel Filterbank Matrix ({n_mels} mels)')
    plt.ylabel('Chỉ số Mel Filter (Mel filter index)')
    plt.xlabel('Tần số (Hz)')
    plt.tight_layout()
    
    matrix_path = os.path.join(output_dir, 'mel_filterbank_matrix.png')
    plt.savefig(matrix_path, dpi=300)
    plt.close()
    print(f"Đã lưu hình ma trận lọc Mel tại: {matrix_path}")
    
    # --- Hình 2: Vẽ đồ thị biên độ của từng bộ lọc (Trực quan sự chồng lấp) ---
    plt.figure(figsize=(12, 6))
    
    # Tạo trục tần số (từ 0 đến tần số Nyquist sr/2)
    frequencies = np.linspace(0, sr / 2, int(1 + n_fft // 2))
    
    # Ve một số bộ lọc đại diện (vẽ cách nhau mỗi 15 dải để đỡ rối mắt)
    for i in range(0, n_mels, 15):
        plt.plot(frequencies, mel_filters[i], label=f'Kiểu lọc số {i}')
        # Fill màu để dễ nhìn sự giao thoa
        plt.fill_between(frequencies, mel_filters[i], alpha=0.2)
        
    plt.title('Biên độ các dải lọc Mel tiêu biểu (Frequency Response)')
    plt.xlabel('Tần số (Hz)')
    plt.ylabel('Trọng số (Weight)')
    plt.xlim([0, sr/2]) # Giới hạn hiện từ 0 tới tần số Nyquist (8000 Hz)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    waves_path = os.path.join(output_dir, 'mel_individual_filters.png')
    plt.savefig(waves_path, dpi=300)
    plt.close()
    print(f"Đã lưu hình dạng sóng các dải lọc tại: {waves_path}")

    # --- Hình 3: Illustration of 128 triangular Mel-Filterbanks overlapping (Như yêu cầu của người dùng) ---
    plt.figure(figsize=(14, 6))
    
    # Vẽ TẤT CẢ 128 dải lọc
    for i in range(n_mels):
        plt.plot(frequencies, mel_filters[i], alpha=0.8, linewidth=1)
        
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.xlim([0, sr / 2])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    all_128_path = os.path.join(output_dir, 'mel_128_triangular_filterbanks.png')
    plt.savefig(all_128_path, dpi=300)
    plt.close()
    print(f"Đã lưu hình 128 Mel-Filterbanks tại: {all_128_path}")

if __name__ == '__main__':
    visualize_mel_filterbanks()
