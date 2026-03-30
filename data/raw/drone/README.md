# Drone Audio Samples (10 giây)

Folder này chứa các audio samples của tiếng drone từ GitHub hoặc nguồn ghi âm khác.

## Đặc điểm file audio:

- **Format**: WAV
- **Sample Rate**: 44100 Hz
- **Duration**: 10 giây (hoặc dài hơn)
- **Loại âm thanh**: Tiếng drone bay (propeller noise, motor sound)

## Quy tắc đặt tên:

- Pattern: `DRONE_001.wav`, `DRONE_002.wav`, `DRONE_003.wav`, ...
- Số thứ tự 3 chữ số (001, 002, 003, ...)

## Nguồn download:

1. **GitHub**: Drone audio datasets
2. **YouTube**: Drone flying videos (extract audio)
3. **Freesound.org**: Search "drone flying", "quadcopter", "UAV"
4. **Ghi âm tự thu**: Dùng điện thoại/microphone ghi âm drone thực tế

## Xử lý sau khi download:

Các file này sẽ được xử lý bởi `src/common/processor.py`:

- Chuẩn hóa (normalization)
- Loại bỏ khoảng lặng (silence removal)
- Trích xuất mel-spectrogram (128 mel bands)
- Cắt thành các đoạn 1 giây → lưu vào `data/processed/drone/`

## Hiện tại:

- ✓ 30 files DRONE (DRONE_001.wav đến DRONE_030.wav)
- Tổng cộng: ~300 giây audio drone

## Thêm file mới:

1. Download/ghi âm file audio drone
2. Convert sang WAV 44100 Hz (nếu cần):
   ```bash
   ffmpeg -i input.mp4 -ar 44100 -ac 1 output.wav
   ```
3. Đổi tên theo pattern: DRONE_031.wav, DRONE_032.wav, ...
4. Copy vào folder này
5. Chạy script cập nhật metadata:
   ```bash
   python scripts/update_metadata.py
   ```
