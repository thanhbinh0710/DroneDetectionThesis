# Kỹ Thuật Sliding Window và Hanning Window

Dự án sử dụng cơ chế cửa sổ (windowing) ở cả cấp độ tiền xử lý tín hiệu, phân chia dữ liệu và hậu xử lý kết quả dự đoán. Dưới đây là chi tiết về các kỹ thuật được áp dụng.

## 1. Kỹ Thuật Sliding Window (Cửa sổ trượt)

Kỹ thuật này được vận dụng ở 2 công đoạn khác nhau với 2 mục đích riêng biệt:

### 1.1. Cắt Dữ liệu Âm thanh (Audio Segmentation)

Cơ chế này (`segment_audio` trong `data_loader.py`) dùng để cắt các đoạn âm thanh dài thành nhiều phân đoạn ngắn để đưa vào huấn luyện hoặc nhận diện:

- **Kích thước cửa sổ (Segment duration)**: `1.0` giây (tương đương 16,000 samples với `sr = 16000`).
- **Độ chồng lấp (Overlap)**: `0.5` (50%).
- **Bước trượt (Hop)**: Cửa sổ sẽ trượt đi `0.5` giây mỗi lần cắt.
- **Mô phỏng**: Với một file audio 2 giây, hệ thống sẽ cắt ra 3 đoạn:
  - Đoạn 1: `0.0s` đến `1.0s`
  - Đoạn 2: `0.5s` đến `1.5s` (trượt đi 0.5s, chứa 50% dữ liệu cũ)
  - Đoạn 3: `1.0s` đến `2.0s`

### 1.2. Lọc Nhiễu Mạng Dự Đoán (Majority Voting)

Cơ chế này (`_register_vote` trong `threads.py`) hoạt động trên chuỗi kết quả inference theo thời gian thực nhằm chống lại sự nhiễu do AI dự đoán sai trong tích tắc (False Positives):

- **Kích thước cửa sổ (Window size)**: `3` (Luôn duy trì và nhìn vào kết quả của 3 phân đoạn audio gần nhất).
- **Ngưỡng chốt (Threshold)**: `2` (Trong 3 phân đoạn đó, phải có ít nhất 2 phân đoạn dự đoán `DRONE` thì mới bật báo động).
- **Mô phỏng**:
  - Trạng thái 1: Lịch sử `[Không, Không, Có]` $\rightarrow$ Tổng số "Có" = 1 (Chưa đạt an toàn $\ge 2$) $\rightarrow$ Báo An toàn.
  - Trạng thái 2: Lịch sử `[Không, Có, Có]` $\rightarrow$ Tổng số "Có" = 2 (Đạt chuẩn quy tắc 2/3) $\rightarrow$ Báo CÓ DRONE.

---

## 2. Cửa Sổ Hanning (Hanning Window) trong STFT

Khi biến đổi âm thanh sang phổ Mel-spectrogram (`librosa.feature.melspectrogram` trong `processor.py`), thư viện sử dụng mặc định cửa sổ **Hanning** trước khi tiến hành Fast Fourier Transform (FFT).

### Tại sao lại cần Hanning Window?

- **Hiện tượng rò rỉ phổ (Spectral Leakage)**: Việc cắt rời âm thanh thành từng khung nhỏ `n_fft = 2048` sẽ tạo ra các điểm gãy gập đột ngột ở hai mép cắt (Start và End). Biến đổi toán học Fourier sẽ hiểu lầm góc gãy này là các âm thanh tần số siêu cao sinh ra nhiễu rác trên đồ thị phổ spectrogram.
- **Cách Hanning xử lý**: Khung audio 2048 mẫu đó được nhân với đồ thị chuông cong của cửa sổ Hanning. Âm thanh ở tâm khung được giữ nguyên, dần vuốt giảm mượt mà về $0$ tại 2 đầu mút mép cắt. Khung âm thanh kết quả không còn nếp gãy và ảnh Mel-spectrogram trích xuất trở nên rõ nguyên bản.

### Sự bù trừ với Overlap (Độ chồng lấp)

Vì Hanning Window triệt tiêu biên độ ở 2 đầu mép khung, một lượng tín hiệu hữu ích ở dải thời gian này sẽ bị che khuất. Tuy nhiên, nó được bù đắp hoàn hảo bởi thông số `hop_length` của STFT:

- **Khung STFT (`n_fft`)**: `2048`
- **Bước nhảy (`hop_length`)**: `512`
- **Hiệu ứng**: Cứ sau `512` mẫu thì một khung mới lại đè lên `2048` mẫu tiếp theo. Sự chênh lệnh này tạo ra sự **chồng lấp tới 75%** (1536 mẫu).
- Nhờ chồng lấp cực dày, phần tín hiệu bị mờ ở biên khung trước lập tức rơi vào khu vực rõ nét ở phần tâm của khung tiếp sau, đảm bảo mọi chi tiết tiềng ồn động cơ Drone đều được chuyển trọn vẹn lên phổ tần số mà không bị đánh rớt.
