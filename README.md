# He thong phat hien va dinh vi drone bang am thanh

He thong phat hien drone dua tren phan tich am thanh voi giao dien PyQt6 va mo hinh CNN. Ung dung ho tro nhan audio tu UDP stream (raw PCM) va chay nhan dien theo thoi gian thuc.

## Tinh nang

- Phat hien drone tu am thanh va hien thi do tin cay
- Dashboard real time (PyQt6 + pyqtgraph)
- Tien xu ly am thanh: normalize, silence removal, mel-spectrogram
- Audio segmentation va data augmentation cho train
- Dau vao UDP audio (raw PCM 16-bit mono, 16 kHz)

## Cau truc du an

```
GUI-ML-Project/
├── data/
│   ├── raw/                     # File am thanh goc (.wav)
│   ├── processed/               # Du lieu da xu ly (.npy)
│   └── metadata.csv             # Thong tin file
├── models/
│   ├── drone_model_v1.keras     # Model CNN da train
│   └── README.md
├── scripts/
│   ├── add_background_samples.py
│   ├── count_samples.py
│   ├── count_samples_simple.py
│   └── test_segmentation.py
├── src/
│   ├── app/
│   │   ├── main.py              # Entry point
│   │   ├── threads.py           # DataWorker (UDP stream + inference)
│   │   ├── hardware.py          # UdpAudioSource
│   │   └── ui/
│   ├── common/
│   │   └── processor.py         # Tien xu ly va mel-spectrogram
│   └── training/
│       ├── data_loader.py
│       └── train.py
├── styles/
│   └── style.qss
├── udp_test2                    # Script UDP test (tham khao giao thuc)
└── README.md
```

## Cai dat

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Su dung

### Chay dashboard

```bash
python -m src.app.main
```

### UDP audio input

Ung dung lang nghe UDP tren port 5555 va nhan raw PCM tu thiet bi:

- Format: PCM 16-bit little-endian
- Channels: 1 (mono)
- Sample rate: 16000 Hz
- Packet size: 1024 frames (2048 bytes)

Trong `src/app/threads.py`, audio duoc buffer de tao cua so du lieu, sau do duoc xu ly o 16000 Hz cho pipeline mel-spectrogram. Neu khong co du lieu, dashboard hien "PAUSED".

File `udp_test2` la script UDP test de tham khao giao thuc va kiem tra stream tu mach.

## Train model

```bash
python -m src.training.data_loader
python -m src.training.train
```

## Du lieu

### Dinh dang metadata.csv

```csv
filename;label;source;duration_sec;notes
drone/DRONE_001.wav;DRONE;Recording;Unknown;Sample audio
background/city_traffic_01.wav;NOT_DRONE;freesound;15;City traffic
```

### Tien xu ly

1. Load audio WAV (mau 16000 Hz trong tap train)
2. Normalize amplitude ve [-1, 1]
3. Silence removal (top_db=20)
4. Mel-spectrogram (n_fft=2048, hop_length=512, n_mels=128)
5. Pad/truncate ve 128 time steps

## Scripts ho tro

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
