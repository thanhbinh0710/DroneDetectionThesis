# Hệ thống Phát hiện và Định vị Drone bằng Âm thanh

Hệ thống phát hiện và định vị drone **dựa trên phân tích âm thanh** với giao diện PyQt6, sử dụng Machine Learning (CNN) để phân loại âm thanh drone.

## ⚡ Tính năng

- **Phát hiện drone bằng âm thanh**: Sử dụng phân tích tín hiệu âm thanh để nhận diện drone
- **Giao diện dashboard thời gian thực**: Hiển thị kết quả phát hiện và độ tin cậy
- **Xử lý âm thanh chuyên sâu**:
  - Chuẩn hóa tín hiệu âm thanh
  - Loại bỏ khoảng lặng
  - Trích xuất Mel-Spectrogram
  - **Audio Segmentation**: Cắt audio dài thành đoạn nhỏ với sliding window
- **Data Augmentation**: Tăng cường dữ liệu huấn luyện với:
  - Thêm nhiễu (noise injection)
  - Thay đổi cao độ (pitch shift)
  - Dịch chuyển thời gian (time shift)
- **Biểu đồ theo dõi**: Hiển thị độ tin cậy phát hiện theo thời gian
- **Kiến trúc module hóa**: Tách biệt rõ ràng giữa phần ứng dụng, huấn luyện và tiện ích chung

## 🎯 Phạm vi dự án

Dự án này **chỉ tập trung vào phát hiện drone thông qua âm thanh**, không sử dụng camera, radar hoặc các cảm biến khác. Hệ thống phân tích đặc trưng âm thanh (mel-spectrogram) để nhận diện và định vị drone dựa trên tiếng động cơ đặc trưng.

## 📂 Cấu trúc dự án

```
GUI-ML-Project/
│
├── data/                        # Quản lý dữ liệu âm thanh
│   ├── raw/                     # File âm thanh gốc (.wav)
│   ├── processed/               # Dữ liệu đã xử lý (.npy)
│   └── metadata.csv             # Thông tin file (filename, label, duration)
│
├── models/                      # Lưu trữ model đã huấn luyện
│   ├── drone_model_v1.keras     # Model CNN đã train
│   ├── scaler.pkl               # Tham số chuẩn hóa (nếu có)
│   └── README.md                # Hướng dẫn về models
│
├── src/                         # Mã nguồn chính
│   ├── common/                  # Module dùng chung
│   │   ├── __init__.py
│   │   └── processor.py         # Xử lý âm thanh & trích xuất Mel-Spectrogram
│   │
│   ├── training/                # Module huấn luyện model
│   │   ├── __init__.py
│   │   ├── train.py             # Script huấn luyện CNN
│   │   └── data_loader.py       # Đọc và chuẩn bị dữ liệu huấn luyện
│   │
│   └── app/                     # Ứng dụng dashboard (PyQt6)
│       ├── __init__.py
│       ├── main.py              # Entry point - chạy ứng dụng
│       ├── threads.py           # Xử lý đa luồng (real-time processing)
│       ├── hardware.py          # Placeholder cho input âm thanh thời gian thực
│       └── ui/                  # Components giao diện
│           ├── __init__.py
│           └── components.py    # ResultPanel, SystemResultWidget
│
├── styles/                      # Qt stylesheet
│   └── style.qss                # Giao diện CSS cho PyQt6
│
├── assets/                      # Tài nguyên (hình ảnh, icon - nếu có)
│
├── requirements.txt             # Thư viện Python cần thiết
└── README.md                    # File này
```

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd GUI-ML-Project
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## 📖 Sử dụng

### Chạy ứng dụng dashboard

```bash
python -m src.app.main
```

Hoặc từ thư mục `src/app/`:

```bash
python main.py
```

### Test xử lý âm thanh và data augmentation

Để kiểm tra quá trình tiền xử lý âm thanh và các kỹ thuật tăng cường dữ liệu:

```bash
python -m src.common.processor
```

Script này sẽ:

- Đọc file âm thanh mẫu từ `data/raw/`
- Áp dụng tiền xử lý (normalization, silence removal)
- Trích xuất Mel-Spectrogram
- Hiển thị 4 biểu đồ so sánh:
  - Mel-Spectrogram gốc
  - Augmentation nhẹ (light)
  - Augmentation vừa (medium)
  - Augmentation mạnh (heavy)

### Test audio segmentation

Để kiểm tra tính năng cắt đoạn audio với sliding window:

```bash
python scripts/test_segmentation.py
```

Script này sẽ:

- Load file audio đầu tiên từ `data/raw/`
- Hiển thị cách audio được cắt thành các đoạn nhỏ
- Visualize segments với matplotlib (2 biểu đồ)
- In chi tiết từng segment (thời gian bắt đầu/kết thúc)

### Huấn luyện model (TODO)

```bash
python -m src.training.train
```

**Lưu ý**: Module huấn luyện hiện chỉ có skeleton code. Cần implement:

- Đọc dữ liệu từ `data/raw/` và `metadata.csv`
- Xây dựng kiến trúc CNN cho phân loại âm thanh
- Huấn luyện với data augmentation
- Lưu model vào `models/`

## 🔊 Quy trình xử lý âm thanh

1. **Thu thập âm thanh**: File .wav với sample rate 44100 Hz
2. **Tiền xử lý**:
   - Normalization (chuẩn hóa biên độ về [-1, 1])
   - Silence removal (loại bỏ khoảng lặng dưới 20dB)
3. **Phân đoạn âm thanh** (tùy chọn):
   - Cắt audio dài thành các đoạn nhỏ với sliding window
   - Ví dụ: Audio 10s → 19 đoạn 1s (overlap 50%)
   - Tăng số lượng mẫu training và tận dụng toàn bộ dữ liệu
4. **Trích xuất đặc trưng**:
   - Mel-Spectrogram (128 mel bands, n_fft=2048, hop_length=512)
   - Chuyển đổi sang thang log-power (dB)
5. **Data Augmentation** (khi huấn luyện):
   - Thêm nhiễu trắng (white noise)
   - Pitch shifting (mô phỏng thay đổi tần số động cơ)
   - Time shifting (dịch chuyển thời gian)
6. **Phân loại**: CNN model → DRONE / NOT_DRONE + confidence score

## ✂️ Audio Segmentation (Phân đoạn âm thanh)

### Tại sao cần phân đoạn audio?

**Vấn đề**: Khi có file audio dài (ví dụ 10 giây), code cũ chỉ lấy phần đầu và bỏ đi phần còn lại.

**Giải pháp**: Sử dụng **sliding window** để cắt audio thành nhiều đoạn nhỏ chồng lấp nhau:

- ✅ Tăng số lượng mẫu training (data augmentation)
- ✅ Tận dụng toàn bộ dữ liệu audio (không bỏ phí)
- ✅ Model học được đặc trưng ở nhiều vị trí khác nhau trong file

### Ví dụ minh họa

**Audio dài 10 giây** với cấu hình:

- `segment_duration=1.0` (mỗi đoạn 1 giây)
- `overlap=0.5` (chồng lấp 50%)

```
Original: [0s -------- 10s]

Segment 1:  [0.0s - 1.0s]
Segment 2:      [0.5s - 1.5s]    ← chồng lấp 50%
Segment 3:          [1.0s - 2.0s]
Segment 4:              [1.5s - 2.5s]
...
Segment 19:                     [9.0s - 10.0s]
```

→ **Kết quả**: 1 file 10s → 19 segments (tăng 19 lần dữ liệu!)

### Cách sử dụng

#### 1. Test tính năng segmentation

Chạy script demo để xem visualization:

```bash
python scripts/test_segmentation.py
```

Script này sẽ:

- Load file audio đầu tiên trong `data/raw/`
- Hiển thị cách audio được cắt thành segments
- Visualize các đoạn với matplotlib

#### 2. Train model với segmentation

**Cách 1**: Không segmentation (mặc định - code cũ)

```python
from src.training.data_loader import load_audio_dataset

X, y, filenames = load_audio_dataset(
    data_dir="data/raw",
    metadata_path="data/metadata.csv"
)
```

**Cách 2**: CÓ segmentation (KHUYẾN NGHỊ cho audio dài)

```python
X, y, filenames = load_audio_dataset(
    data_dir="data/raw",
    metadata_path="data/metadata.csv",
    use_segmentation=True,          # Bật segmentation
    segment_duration=1.0,            # Mỗi đoạn 1 giây
    segment_overlap=0.5              # Chồng lấp 50%
)
```

**Cách 3**: Segmentation + Augmentation (Tăng dữ liệu tối đa)

```python
X, y, filenames = load_audio_dataset(
    data_dir="data/raw",
    metadata_path="data/metadata.csv",
    use_segmentation=True,
    segment_duration=1.0,
    segment_overlap=0.5,
    augment=True,                    # Bật augmentation
    augment_factor=3                 # 3 phiên bản mỗi segment
)
# Kết quả: 1 file 10s → 19 segments → 19*3 = 57 samples!
```

### Tham số cấu hình

| Tham số            | Mô tả                     | Giá trị khuyến nghị |
| ------------------ | ------------------------- | ------------------- |
| `segment_duration` | Độ dài mỗi đoạn (giây)    | `1.0` (1 giây)      |
| `segment_overlap`  | Tỷ lệ chồng lấp (0.0-1.0) | `0.5` (50%)         |

**Lưu ý**:

- Overlap càng cao → càng nhiều segments → dữ liệu nhiều hơn nhưng tương quan cao
- Overlap 0.5 (50%) là giá trị cân bằng tốt
- Với audio ngắn (< 2s), không cần segmentation

## 🧪 Kiến trúc hệ thống

### src/app - Ứng dụng Dashboard

- **main.py**: GUI dashboard với PyQt6
  - Hiển thị kết quả phát hiện âm thanh real-time
  - SystemResultWidget hiển thị kết quả tổng hợp
  - Biểu đồ độ tin cậy theo thời gian
  - Thống kê số lần phát hiện

- **threads.py**: DataWorker thread cập nhật dashboard định kỳ

- **ui/components.py**:
  - `ResultPanel`: Hiển thị trạng thái phát hiện và độ tin cậy
  - `SystemResultWidget`: Kết quả phát hiện tổng hợp

- **hardware.py**: Placeholder cho tích hợp microphone/audio input trong tương lai

### src/common - Module xử lý âm thanh

- **processor.py**: Xử lý âm thanh và trích xuất đặc trưng
  - `preprocess_audio()`: Normalization + silence removal
  - `extract_mel_spectrogram()`: Trích xuất Mel-Spectrogram (128 mel-bands, n_fft=2048)
  - `add_noise()`, `pitch_shift()`, `time_shift()`: Data augmentation

### src/training - Huấn luyện Model

- **train.py**: Script huấn luyện CNN (TODO - cần implement)
- **data_loader.py**: Đọc metadata.csv và trích xuất features từ file âm thanh

## 📊 Dữ liệu

### Định dạng metadata.csv

```csv
filename,label,source,duration_sec,notes
Data_Audio_DRONE_009.wav,DRONE,Recording,Unknown,Sample audio for drone detection testing
```

### Chuẩn bị dữ liệu

1. Đặt file âm thanh (.wav) vào thư mục `data/raw/`
2. Cập nhật file `data/metadata.csv` với thông tin file
3. Chạy `python -m src.training.data_loader` để xử lý và lưu features
4. Features sẽ được lưu vào `data/processed/`

## 🚧 TODO - Các tính năng cần phát triển

- [ ] Hoàn thiện module huấn luyện CNN
- [ ] Tích hợp model đã train vào dashboard (real-time inference)
- [ ] Thêm khả năng ghi âm trực tiếp từ microphone
- [ ] Implement audio streaming processing
- [ ] Thêm visualization cho mel-spectrogram trong dashboard
- [ ] Tối ưu hóa model để chạy nhanh hơn (real-time)
- [ ] Thêm export báo cáo phát hiện

## 📝 Ghi chú kỹ thuật

### Tại sao dùng Mel-Spectrogram?

- Mel-Spectrogram mô phỏng cách tai người nghe âm thanh
- Phù hợp cho phân loại âm thanh với deep learning
- Giảm chiều dữ liệu so với raw audio nhưng vẫn giữ thông tin quan trọng

### Tại sao dùng Data Augmentation?

- Tăng số lượng mẫu huấn luyện từ dữ liệu có hạn
- Giúp model robust hơn với nhiễu và biến đổi trong môi trường thực tế
- Mô phỏng các điều kiện khác nhau: thời tiết, khoảng cách, nhiễu nền

## 📄 License

[Thêm thông tin license nếu cần]

## 👥 Contributors

[Thêm thông tin contributors nếu cần]

- **1**: Chỉ cần 1 sensor
- **2**: Cần ít nhất 2 sensors (mặc định)

## 📊 Data Format

### metadata.csv

```csv
filename,label,source,duration_sec,notes
Data_Audio_DRONE_009.wav,DRONE,Recording,Unknown,Sample audio for testing
```

## 🛠️ Dependencies

- **PyQt6**: GUI framework
- **numpy**: Numerical computing
- **librosa**: Audio processing và feature extraction
- **matplotlib**: Visualization
- **pyqtgraph**: Real-time plotting
- **soundfile**: Audio I/O
- **scikit-learn**: Machine learning utilities
- **pandas**: Data manipulation
- **tensorflow**: Deep learning (for model training)

## 📝 TODO

- [ ] Implement CNN model architecture trong `train.py`
- [ ] Hoàn thiện data augmentation pipeline
- [ ] Tích hợp trained model vào dashboard
- [ ] Load và sử dụng model thực tế thay vì giá trị ngẫu nhiên
- [ ] Thêm model evaluation metrics
- [ ] Export processed features sang data/processed/
- [ ] Thêm logging và error handling
- [ ] Unit tests cho preprocessing functions
- [ ] Tích hợp microphone real-time input
- [ ] Tối ưu hóa performance cho real-time detection

## 👥 Tác giả

[Tên của bạn]

## 📄 License

[Chọn license phù hợp - MIT, Apache 2.0, etc.]

## 🙏 Acknowledgments

- Librosa documentation cho audio preprocessing
- PyQt6 documentation cho GUI development
- Drone audio dataset và research papers về acoustic drone detection
- Drone detection research papers cho methodology
