import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def generate_augmentation_comparison():
    # --- Paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    drone_path = os.path.join(project_root, 'data', 'raw', 'drone', 'DRONE_001.wav')
    noise_path = os.path.join(project_root, 'data', 'raw', 'background', 'BACKGROUND_001.wav')
    output_dir = os.path.join(project_root, 'assets')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'Augmentation_Comparison.png')

    # Parameters
    sr = 16000
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    
    # 1. Load Original Audio
    y_orig, _ = librosa.load(drone_path, sr=sr, duration=2.0)
    
    # 2. Apply Augmentations
    # Pitch Shifting (e.g., +2 semitones)
    y_pitch = librosa.effects.pitch_shift(y_orig, sr=sr, n_steps=2)
    
    # Noise Mixing
    # Try to load a background noise, if not, generate white noise
    try:
        y_noise, _ = librosa.load(noise_path, sr=sr, duration=2.0)
        # Match lengths
        if len(y_noise) < len(y_orig):
            y_noise = np.pad(y_noise, (0, len(y_orig) - len(y_noise)), 'constant')
        else:
            y_noise = y_noise[:len(y_orig)]
    except Exception:
        y_noise = np.random.randn(len(y_orig))
        
    # Mix (Signal to Noise ratio handling simplified: alpha blending)
    # y = y_pitch + 0.1 * y_noise
    noise_factor = 0.05
    y_aug = y_pitch + (noise_factor * y_noise / np.max(np.abs(y_noise)))
    
    # Ensure audio doesn't clip
    if np.max(np.abs(y_aug)) > 1.0:
        y_aug = y_aug / np.max(np.abs(y_aug))

    # 3. Generate Mel-spectrograms
    def get_mel_spec(y):
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB

    S_orig_dB = get_mel_spec(y_orig)
    S_aug_dB = get_mel_spec(y_aug)

    # 4. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original
    img1 = librosa.display.specshow(S_orig_dB, sr=sr, hop_length=hop_length, 
                                    x_axis='time', y_axis='mel', ax=axes[0], cmap='viridis')
    axes[0].set_title('Original Drone Audio')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Frequency (Hz)')
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
    
    # Augmented
    img2 = librosa.display.specshow(S_aug_dB, sr=sr, hop_length=hop_length, 
                                    x_axis='time', y_axis='mel', ax=axes[1], cmap='viridis')
    axes[1].set_title('Augmented (Pitch Shift + Noise)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Đã tạo biểu đồ so sánh Data Augmentation tại: {out_path}")

if __name__ == '__main__':
    generate_augmentation_comparison()
