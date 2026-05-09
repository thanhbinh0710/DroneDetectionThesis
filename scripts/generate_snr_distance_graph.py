import os
import numpy as np
import matplotlib.pyplot as plt

def generate_snr_distance_graph():
    # --- Thông số giả lập thực tế ---
    # Khoảng cách từ 1 mét đến 100 mét
    distances = np.linspace(1, 100, 500) 
    
    # Cường độ âm thanh gốc của UAV (Sound Pressure Level) tại 1 mét (thường ~75 dB cho Flycam nhỏ)
    spl_source = 75 
    
    # Mức ồn môi trường nền (Background noise) - giả sử khu rừng ngoại ô yên tĩnh ~ 45 dB
    noise_level = 45 

    # --- Tính toán theo công thức vật lý Âm học ---
    # Định luật nghịch đảo bình phương: SPL(d) = SPL_0 - 20*log10(d)
    spl_uav = spl_source - 20 * np.log10(distances)
    
    # Ma trận Nhiễu nền không đổi theo khoảng cách
    noise_array = np.full_like(distances, noise_level)

    # --- Vẽ biểu đồ ---
    plt.figure(figsize=(11, 7))

    # Vẽ đường SPL của UAV và Noise
    plt.plot(distances, spl_uav, color='#d62728', linestyle='-', linewidth=3, label='UAV Sound Pressure Level (SPL)')
    plt.plot(distances, noise_array, color='#1f77b4', linestyle='--', linewidth=2.5, label='Environmental Background Noise')

    # Tô màu mô phỏng SNR (Signal-to-Noise Ratio)
    # Vùng có SNR dương (SPL > Noise): Có thể phát hiện được
    plt.fill_between(distances, noise_array, spl_uav, where=(spl_uav >= noise_array), 
                     interpolate=True, color='#2ca02c', alpha=0.2, label='Positive SNR (Acoustically Detectable)')
    
    # Vùng có SNR âm (SPL < Noise): Bị che khuất bởi tiếng ồn môi trường
    plt.fill_between(distances, spl_uav, noise_array, where=(spl_uav < noise_array), 
                     interpolate=True, color='#7f7f7f', alpha=0.3, label='Negative SNR (Masked by Noise)')

    # Điểm giao cắt (SNR = 0 dB)
    intersection_idx = np.argmin(np.abs(spl_uav - noise_array))
    intersect_d = distances[intersection_idx]
    intersect_spl = spl_uav[intersection_idx]
    
    plt.plot(intersect_d, intersect_spl, 'ko', markersize=8, zorder=5)
    plt.annotate(f'SNR = 0 dB\nDetection Limit $\\approx$ {intersect_d:.1f} m', 
                 xy=(intersect_d, intersect_spl), 
                 xytext=(intersect_d + 5, intersect_spl + 10),
                 fontsize=11, fontweight='bold',
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7))

    # --- Định dạng và Tiêu đề ---
    plt.xlabel('Distance from UAV to Microphone (meters)', fontsize=12)
    plt.ylabel('Sound Pressure Level / SPL (dB)', fontsize=12)
    
    plt.xlim([0, 100])
    plt.ylim([30, 80])
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11, loc='upper right')

    plt.tight_layout()

    # --- Lưu tệp ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    output_dir = os.path.join(project_root, 'assets')
    os.makedirs(output_dir, exist_ok=True)
    
    out_path = os.path.join(output_dir, 'SPL_Distance_SNR_Correlation.png')
    plt.savefig(out_path, dpi=300)
    print(f"Đã tạo biểu đồ tại: {out_path}")

if __name__ == '__main__':
    generate_snr_distance_graph()