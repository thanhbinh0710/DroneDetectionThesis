# Processed Drone Features (1 giây segments)

Folder này lưu các đặc trưng đã được trích xuất từ `data/raw/drone/`

## Quy trình xử lý:

1. **Input**: File WAV 10 giây từ `data/raw/drone/DRONE_001.wav`
2. **Processing** (bởi `processor.py`):
   - Chuẩn hóa amplitude → [-1, 1]
   - Loại bỏ silence (threshold 20dB)
   - Cắt thành các đoạn 1 giây (sliding window)
   - Trích xuất mel-spectrogram (128 mel bands × 128 time steps)
   - Data augmentation (noise, pitch shift, time shift)
3. **Output**: File .npy chứa mel-spectrograms

## Cấu trúc dữ liệu:

```
features_drone.npy       # Shape: (N, 128, 128)
                         # N = số lượng segments 1s
                         # 128 mel bands
                         # 128 time steps
```

## Mel-spectrogram Parameters:

- `n_fft`: 2048 (window size)
- `hop_length`: 512 (overlap)
- `n_mels`: 128 (số lượng mel filters)
- `sr`: 44100 Hz

## Augmentation:

Mỗi segment 1s sẽ được tăng cường thành nhiều phiên bản:

- **Original**: Segment gốc
- **Noise**: + Gaussian noise (factor 0.0002-0.001)
- **Pitch shift**: ±2 to ±7 semitones
- **Time shift**: ±20% to ±60% duration

Ví dụ: 10s audio → 10 segments × 4 augmentations = 40 samples

## Tự động tạo:

Folder này được tự động tạo khi chạy:

```bash
python -m src.training.data_loader
```

Không cần edit thủ công!
