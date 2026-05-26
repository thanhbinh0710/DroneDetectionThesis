# Hệ Thống Phát Hiện Và Định Vị UAV (Drone) Bằng Âm Thanh

Dự án máy học phân tích dải âm thanh thời gian thực để nhận diện Drone (UAV) bằng mô hình Mạng Nơ-ron Tích Chập (CNN) kết hợp cùng Giao diện Giám sát Real-time. Hệ thống tiếp nhận luồng âm thanh PCM qua giao thức UDP từ thiết bị ngoại vi, xử lý và hiển thị kết quả lên Dashboard PyQt6.

## Tính Năng Nổi Bật

- **Mô Hình Âm Học (CNN):** Phân loại nhị phân Drone / Background Noise với cấu trúc 4 khối Conv2D (32→64→128→256 filters), tích hợp BatchNormalization, MaxPooling và Dropout.
- **Xử Lý Tín Hiệu Số (DSP):** Chuyển đổi PCM sang Mel-spectrogram (`sr=16000`, `n_fft=2048`, `hop_length=512`, `n_mels=128`, output `128×128×1`).
- **Sliding Window & Temporal Majority Voting:** Cắt luồng âm thanh thành các đoạn 1 giây (overlap 0.5s). Bộ bình chọn đa số theo thời gian (cửa sổ 3 khung, ngưỡng ≥ 2/3) triệt tiêu false positives.
- **Dashboard Real-time:** GUI PyQt6 + pyqtgraph hiển thị Waveform, Confidence Signal, trạng thái phát hiện.
- **Data Augmentation & Nghiên Cứu:** Các script tạo nhiễu (Noise Mixing), dịch tần (Pitch Shifting), xuất biểu đồ phục vụ báo cáo khoa học / luận văn.

## Cách Hoạt Động

```
Thiết bị ghi âm
  │
  ▼  UDP PCM 16-bit @ 16000 Hz (port 5555)
┌─────────────────────────────────────────────────┐
│  Dashboard (PyQt6)                              │
│  ┌───────────────────────────────────────────┐  │
│  │  DataWorker Thread                         │  │
│  │  ┌─────────┐  ┌──────────┐  ┌─────────┐  │  │
│  │  │ Buffer  │→ │ DSP: Mel │→ │ CNN     │  │  │
│  │  │ 1s seg  │  │spectro-  │  │Inference│  │  │
│  │  │ overlap │  │gram      │  │(TF/Keras)│  │  │
│  │  │ 0.5s    │  │128×128×1 │  │         │  │  │
│  │  └─────────┘  └──────────┘  └────┬────┘  │  │
│  │                                   ▼       │  │
│  │                           ┌────────────┐  │  │
│  │                           │ Temporal   │  │  │
│  │                           │ Majority   │→│→│→ Output
│  │                           │ Voting (3  │  │  │   (Cảnh báo +
│  │                           │ frames,    │  │  │   Confidence)
│  │                           │ ≥ 2/3)     │  │  │
│  │                           └────────────┘  │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Cấu Trúc Dự Án

```
GUI-ML-Project/
├── data/
│   ├── raw/                 # Dữ liệu âm thanh gốc (.wav) — drone/, background/
│   ├── processed/           # Feature numpy arrays (.npy) — features, labels, indices
│   └── metadata.csv         # Bảng nhãn, nguồn, thời lượng (sep=';', encoding='utf-8-sig')
├── models/
│   └── drone_model_*.keras  # File CNN đã huấn luyện (tracked via Git LFS)
├── src/
│   ├── app/                 # GUI (PyQt6), Worker Thread, UDP Socket
│   ├── common/              # DSP: preprocess_pcm_audio, extract_mel_spectrogram
│   └── training/            # data_loader (segmentation + augmentation), train (CNN)
├── scripts/
│   ├── count_samples.py               # Thống kê metadata & dữ liệu đã xử lý (pandas)
│   ├── count_samples_fast.py          # Thống kê nhanh (không cần pandas)
│   ├── update_metadata.py             # Tự động cập nhật metadata từ file .wav
│   ├── export_misclassified.py        # Xuất file .wav bị nhận dạng sai để kiểm thử
│   ├── generate_augmentation_comparison.py  # Biểu đồ so sánh Data Augmentation
│   ├── generate_report_spectrograms.py      # Biểu đồ Waveform/Spectrogram cho báo cáo
│   ├── generate_snr_distance_graph.py       # Đồ thị tương quan SPL, Distance, SNR
│   ├── visualize_mel_filters.py       # Biểu đồ phân bố bộ lọc Mel-filterbank
│   ├── predict.py                     # Inference trên file .wav đơn lẻ
│   ├── tune_threshold.py              # Tinh chỉnh ngưỡng phân loại tối ưu
│   ├── rename_background_files.py     # Đổi tên file background hàng loạt
│   ├── test_segmentation.py           # Demo segmentation (sliding window)
│   └── check_duplicates.py            # Kiểm tra trùng lặp trong metadata
├── assets/                 # Đồ họa (.png) xuất từ scripts
├── styles/                 # Qt stylesheet (style.qss)
├── udp_test2/              # Script test UDP cho thiết bị ngoại vi (không thuộc Python project)
├── AGENTS.md               # Command reference cho AI agents
├── ARCHITECTURE_AND_VOTING.md  # Kiến trúc CNN & Voting mechanism
├── MODEL_SPECS.md              # Tham số DSP & Model
├── WINDOW_TECHNIQUES.md        # Kỹ thuật Sliding Window & Hanning
├── DEBUG_CHECKLIST.md          # Troubleshooting hardware/model
└── QUICK_START_DEBUG.md        # Debug checklist tối thiểu
```

## Cài Đặt Khởi Tạo

Yêu cầu: Python 3.9+, Git LFS (để clone model/.npy files)

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt venv (Windows)
venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

## Hướng Dẫn Sử Dụng

### 1. Dashboard (Giao Diện Giám Sát)

```bash
# Chạy Dashboard (lắng nghe UDP port 5555)
python -m src.app.main

# Debug mode (log chi tiết)
$env:DRONE_DEBUG=1; python -m src.app.main
```

Dashboard kết nối UDP với thiết bị ghi âm theo định dạng:
- Định dạng: Raw PCM 16-bit little-endian (Mono)
- Tần số lấy mẫu: 16000 Hz
- Packet Size: 1024 frames (2048 bytes)

### 2. Huấn Luyện Model

```bash
# Bước 1: Trích xuất Mel-spectrogram từ .wav → .npy
python -m src.training.data_loader

# Bước 2: Huấn luyện CNN
python -m src.training.train
# Output: models/drone_model_<timestamp>.keras
```

### 3. Inference trên File

```bash
python scripts/predict.py --audio data/raw/drone/DRONE_001.wav
```

### 4. Scripts Hỗ Trợ Báo Cáo / Luận Văn

```bash
# Xuất biểu đồ (lưu vào assets/)
python -m scripts.generate_report_spectrograms
python -m scripts.generate_snr_distance_graph
python -m scripts.generate_augmentation_comparison
python -m scripts.visualize_mel_filters
```

### 5. Tiện Ích Dữ Liệu

```bash
python -m scripts.count_samples           # Thống kê chi tiết (pandas)
python -m scripts.count_samples_fast      # Thống kê nhanh
python -m scripts.update_metadata         # Cập nhật metadata từ file .wav mới
python -m scripts.export_misclassified    # Xuất file bị phân loại sai
python -m scripts.test_segmentation       # Kiểm thử sliding window
python -m scripts.tune_threshold          # Tìm ngưỡng tối ưu (F1-max)
```

## Real-time Inference (Cơ Chế Chi Tiết)

### Luồng Xử Lý

1. **UDP Receiver:** Nhận gói PCM 16-bit từ thiết bị (port 5555).
2. **Buffer & Segment:** Tích lũy mẫu đến đủ 1 giây (16000 mẫu). Mỗi segment mới chồng lấn 0.5 giây so với segment trước.
3. **DSP:** Chuyển đổi segment thành Mel-spectrogram với các tham số nhất quán xuyên suốt hệ thống (`sr=16000`, `n_fft=2048`, `hop_length=512`, `n_mels=128`).
4. **CNN Inference:** Đưa spectrogram (128×128×1) vào model → output confidence score [0, 1].
5. **Temporal Majority Voting:** Duy trì deque 3 kết quả inference gần nhất. Nếu ≥ 2 trong 3 khung là DRONE (confidence > 0.5), hệ thống báo phát hiện. Khi chưa đủ 3 khung, chỉ cần 1 khung dương là đủ.

### Voting

| Số frame trong cửa sổ | Số frame DRONE cần | Mục đích |
|---|---|---|
| < 3 (khởi động) | ≥ 1 | Phản hồi nhanh lúc ban đầu |
| 3 (ổn định) | ≥ 2 | Lọc false positives, ưu tiên độ ổn định |

## Xử Lý Sự Cố (FAQ)

| Vấn đề | Nguyên nhân & Cách khắc phục |
|---|---|
| **Model không load** | Kiểm tra `MODEL_FILENAME` trong `src/app/threads.py` có khớp với file trong `models/` không. Đảm bảo TensorFlow version khớp. |
| **UDP không nhận dữ liệu** | Kiểm tra port 5555, firewall, địa chỉ IP thiết bị. Dùng Wireshark để debug. |
| **Lỗi oneDNN / TF** | Đặt biến môi trường `TF_ENABLE_ONEDNN_OPTS=0` và `TF_CPP_MIN_LOG_LEVEL=2` (đã được xử lý sẵn trong code). |
| **Pandas đọc metadata lỗi** | Dùng `pd.read_csv('data/metadata.csv', sep=';', encoding='utf-8-sig')`. File dùng delimiter `;` và BOM. |
| **Git LFS** | File `.keras`, `.npy` được track bằng Git LFS. Chạy `git lfs install` và `git lfs pull` sau khi clone. |
| **ImportError: src.*** | Chạy từ thư mục gốc của project. `threads.py` dùng `sys.path.insert(0, ...)` để resolve path. |
| **False positive nhiều** | Thử tăng voting window hoặc dùng `tune_threshold.py` để tìm ngưỡng tối ưu. |

## Tài Liệu Tham Khảo

- [`AGENTS.md`](./AGENTS.md) — Danh sách lệnh đầy đủ, tham số DSP, kiến trúc model, voting mechanism.
- [`ARCHITECTURE_AND_VOTING.md`](./ARCHITECTURE_AND_VOTING.md) — Chi tiết thuật toán biểu quyết và kiến trúc CNN.
- [`MODEL_SPECS.md`](./MODEL_SPECS.md) — Cấu trúc tham số mạng học sâu (CNN).
- [`WINDOW_TECHNIQUES.md`](./WINDOW_TECHNIQUES.md) — Kỹ thuật chia khung và chồng lấn thời gian.
- [`DEBUG_CHECKLIST.md`](./DEBUG_CHECKLIST.md) — Troubleshooting phần cứng & model.
- [`QUICK_START_DEBUG.md`](./QUICK_START_DEBUG.md) — Debug checklist tối thiểu.

## Dependencies

- PyQt6
- numpy
- librosa
- matplotlib
- pyqtgraph
- soundfile
- scikit-learn
- pandas
- tensorflow
