"""
Generate Waveform & Spectrogram figures for thesis/report (Figure 4.3 equivalent).
Usage: python -m scripts.generate_report_spectrograms
"""

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# 1. Load an actual drone audio file from the project's dataset
audio_path = os.path.join(project_root, 'data', 'raw', 'drone', 'DRONE_001.wav') 

print(f"Đang xử lý file: {audio_path}")
y, sr = librosa.load(audio_path, sr=16000, duration=1.0) # Ensure 16kHz sample rate

# Set the figure size
plt.figure(figsize=(10, 8))

# ==========================================
# Figure 4.3a: Time-domain Waveform
# ==========================================
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr, color='blue')
plt.title('a) Waveform of Drone Audio (Time Domain)', fontsize=14)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# ==========================================
# Figure 4.3b: Log-Mel Spectrogram Computation
# Specs from MODEL_SPECS.md
# ==========================================
n_fft = 2048
hop_length = 512
n_mels = 128

# Compute Mel Spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmax=8000)
# Convert to Log-Scale (Decibels)
log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

plt.subplot(2, 1, 2)
# Using 'magma' colormap for high-contrast harmonic visualization
img = librosa.display.specshow(log_mel_spec, sr=sr, hop_length=hop_length, 
                               x_axis='time', y_axis='mel', fmax=8000, cmap='magma')
cbar = plt.colorbar(img, format='%+2.0f dB')
cbar.set_label('Magnitude (dB)', fontsize=12)

plt.title('b) Log-Mel Spectrogram (128x128x1 Input Tensor for CNN)', fontsize=14)
plt.xlabel('Time (seconds)', fontsize=12)
plt.ylabel('Mel Frequency (Hz)', fontsize=12)

# Optimize layout and save the high-resolution image
plt.tight_layout()
output_dir = os.path.join(project_root, 'assets')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'Figure_4_3_Spectrogram_English.png')

plt.savefig(output_path, dpi=300) 
print(f"Đã lưu hình ảnh độ phân giải cao tại: {output_path}")