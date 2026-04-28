# Ví dụ đặt tên file Background Audio

## Co bat buoc phai dat ten "BACKGROUND_001.wav" khong?

**TRẢ LỜI: KHÔNG!** Bạn có thể đặt bất kỳ tên nào miễn là:

1. File có định dạng `.wav`
2. Đặt trong folder `data/raw/background/`
3. Có trong `metadata.csv` với label `NOT_DRONE`

---

## Cac cach dat ten hop le:

### Cách 1: Tên mô tả chi tiết (KHUYẾN NGHỊ)

```
city_traffic_downtown.wav       <- Ro rang, de hieu
rain_heavy_outdoor.wav          <- Mo ta chinh xac
bird_sparrow_morning.wav        <- Chi tiet loai chim
wind_strong_forest.wav          <- Cu the moi truong
airplane_boeing737_takeoff.wav  <- Rat chi tiet
construction_jackhammer.wav     <- Ro nguon goc
office_keyboard_typing.wav      <- Mo ta hoat dong
highway_traffic_night.wav       <- Them ngu canh thoi gian
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

## Ten file nen tranh:

```
file #1.wav              <- Co ky tu dac biet #
background;noise.wav     <- Co dau cham phay ;
am thanh gio.wav         <- Co dau cach va tieng Viet (nen dung: am_thanh_gio.wav)
traffic&wind.wav         <- Co ky tu &
noise@50hz.wav           <- Co ky tu @
file%.wav                <- Co ky tu %
```

**Lý do tránh:**

- Dấu chấm phẩy `;` làm lỗi CSV delimiter
- Ký tự đặc biệt có thể gây lỗi khi đọc file trên OS khác nhau
- Dấu cách có thể gây lỗi trong command line

---

## Vi du metadata.csv voi ten tuy y:

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

## Script tu dong cap nhat metadata

Khi bạn chạy:

```bash
python scripts/add_background_samples.py
```

Script sẽ **TỰ ĐỘNG** tìm TẤT CẢ file `.wav` trong `data/raw/background/` và thêm vào metadata, **KHÔNG quan tâm** tên file là gì!

**Ví dụ output:**

```
Found 8 background audio files:
  1. city_traffic_downtown.wav     (2.34 MB)
  2. rain_heavy_outdoor.wav        (3.12 MB)
  3. bird_sparrow_morning.wav      (1.56 MB)
  4. wind_strong_forest.wav        (1.89 MB)
  5. airplane_boeing737_takeoff.wav (4.01 MB)
  6. construction_jackhammer.wav   (2.78 MB)
  7. office_keyboard_typing.wav    (3.45 MB)
  8. highway_traffic_night.wav     (2.90 MB)

Adding 8 new NOT_DRONE entries to metadata...
Updated metadata saved: metadata.csv
```

---

## Khuyen nghi:

### Dùng tên MÔ TẢ khi:

- Dataset nho (< 100 files)
- Muon de dang identify file
- Chia se dataset voi nguoi khac
- Tao bao cao, presentation

### Dùng tên SỐ THỨ TỰ khi:

- Dataset lon (> 100 files)
- Tu dong download/generate hang loat
- Chi can tracking don gian
- Dung script automation

---

## Tom lai:

| Tiêu chí        | Yêu cầu                                            |
| --------------- | -------------------------------------------------- |
| **Tên file**    | Tùy ý, miễn là rõ ràng và không có ký tự đặc biệt  |
| **Format**      | `.wav` (bắt buộc)                                  |
| **Location**    | `data/raw/background/` (bắt buộc)                  |
| **Metadata**    | Phải có trong `metadata.csv` với label `NOT_DRONE` |
| **Sample Rate** | 44100 Hz (khuyến nghị)                             |
| **Encoding**    | PCM 16-bit hoặc 24-bit                             |

**Chon cach dat ten phu hop voi workflow cua ban!**
