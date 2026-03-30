# Background / Non-Drone Audio Samples

Folder này chứa các audio samples cho class **NOT_DRONE** (background noise).

## Mục đích

- Train binary classifier: DRONE vs NOT_DRONE
- Giúp model học phân biệt âm thanh drone với các âm thanh nền khác

## Loại âm thanh nên thêm vào đây:

1. **Ambient noise**: Tiếng gió, tiếng mưa, tiếng rừng
2. **City sounds**: Tiếng xe cộ, tiếng người đi lại, tiếng xây dựng
3. **Bird sounds**: Tiếng chim hót (dễ nhầm với drone)
4. **Aircraft sounds**: Tiếng máy bay (khác với drone)
5. **Machinery**: Tiếng máy móc, quạt, điều hòa
6. **White noise**: Nhiễu trắng, nhiễu phông

## Quy tắc đặt tên file:

### ✅ TÙY Ý - Có thể đặt bất kỳ tên nào!

Bạn **KHÔNG** bắt buộc phải đặt tên theo pattern cố định. Có thể đặt tên mô tả rõ ràng:

**Ví dụ tên file hợp lệ:**

- ✓ `city_traffic_01.wav` - Rõ ràng, dễ hiểu
- ✓ `rain_sound.wav` - Mô tả nội dung
- ✓ `bird_chirping_park.wav` - Chi tiết nguồn gốc
- ✓ `wind_noise_outdoor.wav` - Cụ thể môi trường
- ✓ `airplane_passing.wav` - Tên mô tả
- ✓ `BACKGROUND_001.wav` - Pattern số thứ tự (nếu muốn)
- ✓ `NON_DRONE_001.wav` - Pattern khác (nếu muốn)

**Yêu cầu:**

- ✅ Format: WAV
- ✅ Sample rate: 44100 Hz (giống DRONE samples)
- ✅ Duration: 5-30 giây
- ✅ Đặt trong folder: `data/raw/background/`
- ✅ Có trong `metadata.csv` với label `NOT_DRONE`

**Lưu ý:**

- ⚠️ Tránh ký tự đặc biệt: `#`, `%`, `&`, `@`, dấu cách → dùng `_` thay thế
- ⚠️ Không dùng dấu chấm phẩy `;` trong tên file

## Cập nhật metadata

Sau khi thêm files vào folder này, cần cập nhật `data/metadata.csv`:

```csv
file_name,label,class_id
DRONE_001.wav,DRONE,1
DRONE_002.wav,DRONE,1
...
BACKGROUND_001.wav,NOT_DRONE,0
BACKGROUND_002.wav,NOT_DRONE,0
...
```

## Sources gợi ý để download background sounds:

- Freesound.org - https://freesound.org/
- BBC Sound Effects - https://sound-effects.bbcrewind.co.uk/
- YouTube Audio Library
- Environmental Sound Classification datasets (ESC-50, UrbanSound8K)

## Số lượng khuyến nghị:

- Tối thiểu: 30 samples (cân bằng với 30 DRONE samples)
- Tối ưu: 50-100 samples cho training tốt hơn
