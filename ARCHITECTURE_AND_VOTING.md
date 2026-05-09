# Kiến trúc Mô hình và Cơ chế Bỏ phiếu (Model Architecture & Voting Mechanism)

Tài liệu này mô tả chi tiết về cơ chế lọc nhiễu dự đoán và kiến trúc mạng nơ-ron tích chập (CNN) được triển khai thực tế trong dự án phát hiện drone bằng âm thanh.

## 1. Cơ chế "Majority Voting" (Bỏ phiếu theo đa số)
Dự án sử dụng cơ chế bỏ phiếu **Trường hợp 1: Theo thời gian (Temporal Voting)** thay vì Ensemble Learning.

* **Cách hoạt động**: Việc bình chọn không đến từ nhiều mô hình khác biệt mà đến từ **1 mô hình nhưng chạy trên các khung thời gian liên tiếp**.
* **Chi tiết cấu hình**: Hệ thống duy trì một cửa sổ (sliding window) kích thước = **3 khung** (tương ứng với 3 phân đoạn audio 1 giây liền kề theo thời gian).
* **Điều kiện chốt ngưỡng**: Phải có ít nhất **2/3** khung trong cửa sổ dự đoán là `DRONE` thì hệ thống mới xuất trạng thái cảnh báo và ghi nhận phát hiện. Việc này tự động triệt tiêu các khoảng nhiễu gây báo động giả (False Positives) chỉ xuất hiện chớp nhoáng trong 1 tích tắc.

## 2. Kiến trúc mạng CNN
Cấu trúc mạng (được định nghĩa trong hàm `build_model` tại `src/training/train.py`) là một kiến trúc mạng nơ-ron sâu chuyên biệt cho thị giác máy tính, giờ áp dụng xử lý ảnh phổ âm thanh:

### 2.1. Lớp trích xuất đặc trưng (Feature Extraction)
Bao gồm **4 khối chập (Convolutional Blocks)** nối tiếp nhau. Mỗi khối hoạt động theo luồng:
1. **Conv2D (kernel 3x3)**: Lần lượt mở rộng số lượng bộ lọc (filters) qua các lớp sâu hơn: **32 $\rightarrow$ 64 $\rightarrow$ 128 $\rightarrow$ 256**.
2. **BatchNormalization**: Chuẩn hóa Batch giúp ổn định các gradient phân giải và tăng tốc độ hội tụ mô hình.
3. **MaxPooling2D (pool size 2x2)**: Giảm kích thước không gian (chiều dài, chiều rộng) theo từng khối.
4. **Dropout (0.25)**: Vô hiệu hóa ngẫu nhiên 25% nơ-ron nhằm chống học vẹt (overfitting).

### 2.2. Lớp phân loại (Classification Head)
* Khối đặc trưng 2D ở cuối sẽ được duỗi thẳng (`Flatten`) thành vector 1 chiều.
* Biến đổi qua **2 lớp nối đầy đủ (Dense / Fully Connected)** với kích thước lần lượt là **256** và **128** units. Cả 2 đều dùng hàm kích hoạt **ReLU** và áp dụng `Dropout(0.5)`.

### 2.3. Lớp Output và Tối ưu hóa
* **Lớp Output**: Chỉ có **1 node** duy nhất do đây là bài toán phân loại nhị phân (Binary Classification: Drone vs. Not_Drone).
* **Hàm kích hoạt Output**: **Sigmoid**. Cung cấp một giá trị xác suất (Confidence) nằm trong khoảng $[0, 1]$. Ngưỡng đánh giá (Threshold) để gán nhãn DRONE đang được thiết lập là $> 0.5$.
* **Hàm mất mát (Loss function)**: **Binary Cross-Entropy** (`binary_crossentropy`).
* **Thuật toán tối ưu (Optimizer)**: **Adam** với `learning_rate = 0.001` kết hợp callback giảm LR (ReduceLROnPlateau) nếu quá trình đo kiểm validation loss không tiến triển.

## 3. Khẳng định về các thuật toán học máy khác
Trong hệ thống thực tế hoàn chỉnh này, dự án **KHÔNG** sử dụng thuật toán Random Forest, SVM hay LSTM.

Toàn bộ quá trình từ đào tạo học máy cho đến giao diện realtime (PyQt6/UDP) đều chỉ sử dụng bộ đôi kết hợp: **Mạng Nơ-ron Tích chập 2 chiều (2D CNN)** qua TensorFlow/Keras cho việc suy luận biểu diễn phổ Mel, đi kèm với **Temporal Majority Voting** giúp liên kết suy luận theo trình tự thời gian (nhằm khắc phục những điểm yếu mà LSTM giải quyết mà không tiêu tốn lượng tính toán quá lớn).