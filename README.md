# Hệ Thống Phát Hiện Và Định Vị UAV (Drone) Bằng Âm Thanh

Dự án máy học phân tích dải âm thanh thời gian thực để nhận diện Drone (UAV) bằng mô hình Mạng Nơ-ron Tích Chập (CNN) kết hợp cùng Giao diện Giám sát Real-time. Xây dựng và liên kết với Hardware thông qua giao thức UDP Stream để tiếp nhận và phân tích raw PCM liên tục.

## Tính Năng Nổi Bật

- **Mô Hình Âm Học (CNN):** Phân loại nhị phân Âm thanh Drone và Background Noise với cấu trúc 4 khối Conv2D (32, 64, 128, 256 filters), tích hợp BatchNormalization, MaxPooling và Dropout.
- **Xử Lý Tín Hiệu Số (DSP):** Chuyển đổi chuỗi thời gian sang dải tần số đặc trưng (Mel-spectrogram) bằng `librosa`. Thông số chuyên sâu: `sr=16000`, `n_fft=2048`, `hop_length=512`, `n_mels=128`.
- **Hệ Vi Phân Cửa Sổ Trượt (Sliding Window & Voting):** Cắt luồng âm thanh thành các đoạn 1 giây, chồng lấn (overlap) 0.5 giây. Cơ chế **Temporal Majority Voting** (chốt kết quả dựa trên $\ge$ 2/3 khung cửa sổ) giúp triệt tiêu nhiễu giả (false positives).
- **Trực Quan Hóa (Dashboard):** Giao diện GUI được xây dựng bằng `PyQt6` và `pyqtgraph`, hỗ trợ hiển thị biểu đồ tần số âm thanh (Waveform) và biểu đồ đo lường độ tin cậy (Confidence Signal).
- **Hỗ Trợ Data Augmentation & Nghiên Cứu:** Các script tạo dữ liệu nhiễu (Noise Mixing), dịch tần (Pitch Shifting), và biểu đồ đồ họa hỗ trợ đưa vào báo cáo khoa học/luận văn.

## Cấu Trúc Dự Án

```
GUI-ML-Project/
├── data/
│   ├── raw/                 # Dữ liệu âm thanh gốc (.wav) - Drone & Background
│   ├── processed/           # Dữ liệu mảng numpy đã trích xuất đặc trưng (.npy)
│   └── metadata.csv         # Bảng nhãn, nguồn, thời lượng của file âm thanh
├── models/
│   └── drone_model_v1.keras # File Model CNN đã huấn luyện & test thành công
├── src/
│   ├── app/                 # UI và Luồng chính (GUI, Threads, UDP Socket)
│   ├── common/              # Module tiền xử lý DSP & Feature Extraction
│   └── training/            # Pipeline Data Loader & Build / Train Model
├── scripts/
│   ├── count_samples*.py                # Đếm và thống kê dữ liệu
│   ├── export_misclassified_audio.py    # Kiểm thử và xuất các file bị nhận dạng sai
│   ├── generate_augmentation_comparison.py # Render biểu đồ so sánh Data Augmentation
│   ├── generate_figure_4_3.py           # Render biểu đồ Waveform/Spectrogram cho báo cáo
│   ├── generate_snr_distance_graph.py   # Render đồ thị tương quan SPL, Distance và SNR
│   └── visualize_mel_filters.py         # Render biểu đồ phân bố bộ lọc Mel-filterbank
├── assets/                  # Lưu trữ các file đồ họa (.png) xuất ra từ scripts
├── styles/                  # File giao diện (CSS/QSS)
├── udp_test2                # Script UDP test thiết bị ngoại vi
└── *.md                     # Tài liệu kỹ thuật (ARCHITECTURE, MODEL_SPECS, v.v.)
```

## Cài Đặt Khởi Tạo

Yêu cầu: Python 3.9+

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt venv (Windows)
venv\Scripts\activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

## Hướng Dẫn Sử Dụng

### 1. Khởi chạy Giao Diện Giám Sát (Dashboard)

```bash
python -m src.app.main
```

Ứng dụng sẽ mở port `5555` lắng nghe thiết bị truyền luồng:

- Định dạng: Raw PCM 16-bit little-endian (Mono)
- Tần số lấy mẫu: 16000 Hz
- Packet Size: 1024 frames (2048 bytes)

### 2. Huấn Luyện (Training) Model

Tiến hành load dữ liệu từ raw sang mel-spectrogram, sau đó huấn luyện mô hình:

```bash
# B1: Trích xuất đặc trưng từ âm thanh (.wav -> .npy)
python -m src.training.data_loader

# B2: Cấu trúc hóa mạng nơ-ron và tiến hành Train
python -m src.training.train
```

### 3. Kết xuất Đồ Họa & Báo Cáo

Nếu bạn đang thực hiện đồ án/luận văn, các tệp ở thư mục `scripts/` hỗ trợ vẽ biểu đồ bằng `matplotlib` với độ phân giải cực cao (300 DPI):

```bash
python -m scripts.generate_snr_distance_graph
python -m scripts.generate_figure_4_3
python -m scripts.generate_augmentation_comparison
# Ảnh sẽ tự động lưu vào /assets/
```

## Tài Liệu Tham Khảo (Docs)

Dự án đi kèm các tài liệu phân tích sâu bên trong mã nguồn:

- [`ARCHITECTURE_AND_VOTING.md`](./ARCHITECTURE_AND_VOTING.md) - Chi tiết Thuật toán Biểu quyết, Sliding Window.
- [`MODEL_SPECS.md`](./MODEL_SPECS.md) - Cấu trúc tham số Mạng học sâu (CNN).
- [`WINDOW_TECHNIQUES.md`](./WINDOW_TECHNIQUES.md) - Kỹ thuật Chia khung và Chồng lấn thời gian.

- `scripts/count_samples.py`: thong ke metadata va du lieu da xu ly
- `scripts/count_samples_simple.py`: thong ke nhanh (khong can pandas)
- `scripts/add_background_samples.py`: tu dong cap nhat metadata
- `scripts/test_segmentation.py`: demo segmentation

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
