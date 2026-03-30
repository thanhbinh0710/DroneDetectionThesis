# Processed Background Features (1 giây segments)

Folder này lưu các đặc trưng đã được trích xuất từ `data/raw/background/`

## Quy trình xử lý:

1. **Input**: File WAV từ `data/raw/background/BACKGROUND_001.wav`
2. **Processing** (bởi `processor.py`):
   - Chuẩn hóa amplitude → [-1, 1]
   - Loại bỏ silence (threshold 20dB)
   - Cắt thành các đoạn 1 giây (sliding window)
   - Trích xuất mel-spectrogram (128 mel bands × 128 time steps)
   - Data augmentation (noise, pitch shift, time shift)
3. **Output**: File .npy chứa mel-spectrograms

## Cấu trúc dữ liệu:

```
features_background.npy  # Shape: (N, 128, 128)
                         # N = số lượng segments 1s
                         # 128 mel bands
                         # 128 time steps
```

## Tại sao cần background samples?

- **Binary Classification**: Model cần học phân biệt DRONE vs NOT_DRONE
- **Reduce False Positives**: Tránh nhầm tiếng gió, xe cộ, chim hót với drone
- **Robust Model**: Model hoạt động tốt trong môi trường thực tế

## Loại background nên có:

✓ City traffic (xe cộ)
✓ Wind and rain (gió, mưa)
✓ Birds chirping (chim hót)
✓ Airplane/helicopter (máy bay, trực thăng)
✓ Construction/machinery (xây dựng, máy móc)
✓ White noise (nhiễu trắng)
✓ Office ambient (văn phòng)
✓ Nature sounds (tự nhiên)

## Dataset Balance:

Để model học tốt, nên cân bằng:

- 30 DRONE samples → 30 BACKGROUND samples
- 50 DRONE samples → 50 BACKGROUND samples

## Tự động tạo:

Folder này được tự động tạo khi chạy:

```bash
python -m src.training.data_loader
```

Sau khi đã thêm background audio vào `data/raw/background/`
