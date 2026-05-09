# Thông Số Kỹ Thuật Mô Hình (Model Specifications)

Dưới đây là các thông số cụ thể của mô hình nhận diện âm thanh drone đang sử dụng trong dự án:

## 1. Thông số tín hiệu đầu vào (Audio Input)

- **Tần số lấy mẫu (Sample Rate - $f_{sample}$)**: Toàn hệ thống (từ tiền xử lý dữ liệu huấn luyện, đầu vào UDP đến lúc inference) được đồng nhất là **16000 Hz (16 kHz)**.
- **Độ dài mỗi phân đoạn (Window/Chunk length)**: Âm thanh được cắt thành các chunk (phân đoạn) có độ dài **1.0 giây**.
  - Trong pipeline huấn luyện (`data_loader.py`): overlap là `0.5` (chồng lấp 50%).
  - Trong realtime inference (`threads.py`): audio buffer size được đảm bảo lưu trữ lượng dữ liệu bằng đúng 1 giây (`16000 Hz * 2 bytes * 1.0 s = 32000 bytes`).

## 2. Thông số biến đổi STFT (Short-Time Fourier Transform)

_(Cấu hình tại hàm `extract_mel_spectrogram` trong `processor.py`)_

- **Kích thước cửa sổ FFT (`n_fft`)**: **2048**
- **Bước nhảy (`hop_length`)**: **512** (~32ms mỗi lần trượt so với đoạn audio 1 giây)
- **Hàm cửa sổ (Window function)**: Hệ thống sử dụng **hann window (Hanning)** — là thông số mặc định của hàm `librosa.feature.melspectrogram`.

## 3. Thông số dải lọc Mel (Mel-Filterbanks)

- **Số lượng dải Mel (`n_mels`)**: **128 dải**.
- **Giới hạn tần số (`fmin` và `fmax`)**: Không cấu hình giới hạn (dùng mặc định toàn dải). Toàn dải tần Nyquist (**0 - 8000 Hz** cho âm thanh 16kHz) sẽ được lấy làm mốc chuẩn.

## 4. Định dạng đầu ra cho mô hình CNN

- **Kích thước đầu vào mạng CNN (Input Shape)**: Mật độ khung hình được chuẩn hóa qua `pad_or_truncate_spectrogram`. Kích thước cuối cùng đưa vào mạng lưới CNN 2D là tensor: **128 $\times$ 128 $\times$ 1** ($128$ dải mels và $128$ time steps).
  _(Chi tiết: 1 giây âm thanh với `hop_length` là 512 sinh ra khoảng 32 frames (16000/512 = 31.25). Sau đó mảng này được padding thêm Zero-Padding để đạt kích thước cố định là 128 time steps.)_
- **Kỹ thuật chuẩn hóa (Normalization)**:
  - **Thang biên độ âm thanh (Amplitude Normalization)**: Chuẩn hóa tín hiệu gốc về khoảng **[-1, 1]** thông qua hàm `librosa.util.normalize`, và phép chia dải giá trị thập phân tự động (`/ 32768.0`) cho kiểu số nguyên 16-bit khi truyền raw PCM.
  - **Thang Mel Spectrogram (Decibel Normalization)**: Chuyển đổi Mel Power về không gian Log-Scale (Decibel) bằng dòng lệnh `librosa.power_to_db(mel_spec, ref=np.max)`. Điều này scale toàn bộ đầu vào trong đó định vị đỉnh 0 dB và các vùng không có năng lượng hiển thị với số âm lớn.
