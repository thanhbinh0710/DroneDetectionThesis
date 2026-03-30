# Ví dụ đặt tên file Background Audio

## ❓ Có bắt buộc phải đặt tên "BACKGROUND_001.wav" không?

**TRẢ LỜI: KHÔNG!** Bạn có thể đặt bất kỳ tên nào miễn là:

1. File có định dạng `.wav`
2. Đặt trong folder `data/raw/background/`
3. Có trong `metadata.csv` với label `NOT_DRONE`

---

## ✅ Các cách đặt tên HỢP LỆ:

### Cách 1: Tên mô tả chi tiết (KHUYẾN NGHỊ)

```
city_traffic_downtown.wav       ← Rõ ràng, dễ hiểu
rain_heavy_outdoor.wav          ← Mô tả chính xác
bird_sparrow_morning.wav        ← Chi tiết loại chim
wind_strong_forest.wav          ← Cụ thể môi trường
airplane_boeing737_takeoff.wav  ← Rất chi tiết
construction_jackhammer.wav     ← Rõ nguồn gốc
office_keyboard_typing.wav      ← Mô tả hoạt động
highway_traffic_night.wav       ← Thêm ngữ cảnh thời gian
```

### Cách 2: Pattern số thứ tự đơn giản

```
BACKGROUND_001.wav
BACKGROUND_002.wav
BACKGROUND_003.wav
...
BACKGROUND_030.wav
```

### Cách 3: Kết hợp prefix + mô tả

```
BG_traffic_01.wav
BG_rain_02.wav
BG_birds_03.wav
NON_DRONE_wind_01.wav
NON_DRONE_airplane_02.wav
```

### Cách 4: Tên gốc từ nguồn download (nếu rõ ràng)

```
freesound_12345_traffic.wav
esc50_dog_barking.wav
urbansound_siren.wav
youtube_rain_sound_hd.wav
```

---

## ❌ Tên file NÊN TRÁNH:

```
❌ file #1.wav              ← Có ký tự đặc biệt #
❌ background;noise.wav     ← Có dấu chấm phẩy ;
❌ âm thanh gió.wav         ← Có dấu cách và tiếng Việt (nên dùng: am_thanh_gio.wav)
❌ traffic&wind.wav         ← Có ký tự &
❌ noise@50hz.wav           ← Có ký tự @
❌ file%.wav                ← Có ký tự %
```

**Lý do tránh:**

- Dấu chấm phẩy `;` làm lỗi CSV delimiter
- Ký tự đặc biệt có thể gây lỗi khi đọc file trên OS khác nhau
- Dấu cách có thể gây lỗi trong command line

---

## 📋 Ví dụ metadata.csv với tên tùy ý:

```csv
filename;label;source;duration_sec;notes
drone/DRONE_001.wav;DRONE;Recording;Unknown;Drone sample 1
drone/DRONE_002.wav;DRONE;Recording;Unknown;Drone sample 2
background/city_traffic_downtown.wav;NOT_DRONE;freesound;15;Busy city traffic
background/rain_heavy_outdoor.wav;NOT_DRONE;youtube;20;Heavy rain sound
background/bird_sparrow_morning.wav;NOT_DRONE;recorded;10;Morning bird chirping
background/wind_strong_forest.wav;NOT_DRONE;freesound;12;Strong wind in forest
background/airplane_boeing737_takeoff.wav;NOT_DRONE;youtube;25;Commercial airplane
background/construction_jackhammer.wav;NOT_DRONE;recorded;18;Construction site
background/office_keyboard_typing.wav;NOT_DRONE;recorded;30;Office ambient
background/highway_traffic_night.wav;NOT_DRONE;freesound;22;Highway at night
```

---

## 🔧 Script tự động cập nhật metadata

Khi bạn chạy:

```bash
python scripts/add_background_samples.py
```

Script sẽ **TỰ ĐỘNG** tìm TẤT CẢ file `.wav` trong `data/raw/background/` và thêm vào metadata, **KHÔNG quan tâm** tên file là gì!

**Ví dụ output:**

```
✓ Found 8 background audio files:
  1. city_traffic_downtown.wav     (2.34 MB)
  2. rain_heavy_outdoor.wav        (3.12 MB)
  3. bird_sparrow_morning.wav      (1.56 MB)
  4. wind_strong_forest.wav        (1.89 MB)
  5. airplane_boeing737_takeoff.wav (4.01 MB)
  6. construction_jackhammer.wav   (2.78 MB)
  7. office_keyboard_typing.wav    (3.45 MB)
  8. highway_traffic_night.wav     (2.90 MB)

📝 Adding 8 new NOT_DRONE entries to metadata...
✓ Updated metadata saved: metadata.csv
```

---

## 💡 Khuyến nghị:

### Dùng tên MÔ TẢ khi:

- ✅ Dataset nhỏ (< 100 files)
- ✅ Muốn dễ dàng identify file
- ✅ Chia sẻ dataset với người khác
- ✅ Tạo báo cáo, presentation

### Dùng tên SỐ THỨ TỰ khi:

- ✅ Dataset lớn (> 100 files)
- ✅ Tự động download/generate hàng loạt
- ✅ Chỉ cần tracking đơn giản
- ✅ Dùng script automation

---

## 🎯 TÓM LẠI:

| Tiêu chí        | Yêu cầu                                            |
| --------------- | -------------------------------------------------- |
| **Tên file**    | Tùy ý, miễn là rõ ràng và không có ký tự đặc biệt  |
| **Format**      | `.wav` (bắt buộc)                                  |
| **Location**    | `data/raw/background/` (bắt buộc)                  |
| **Metadata**    | Phải có trong `metadata.csv` với label `NOT_DRONE` |
| **Sample Rate** | 44100 Hz (khuyến nghị)                             |
| **Encoding**    | PCM 16-bit hoặc 24-bit                             |

✨ **Chọn cách đặt tên phù hợp với workflow của bạn!**
