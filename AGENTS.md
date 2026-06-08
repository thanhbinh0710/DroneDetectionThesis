# AGENTS.md — Drone Audio Detection

## Commands

```bash
# Dashboard (GUI, listens UDP :5555)
python -m src.app.main

# Debug mode
$env:DRONE_DEBUG=1; python -m src.app.main

# Training pipeline (must run in order)
python -m src.training.data_loader   # .wav → data/processed/features.npy
python -m src.training.train         # train → models/drone_model_*.keras

# Script-based inference on a file
python scripts/predict.py --audio path/to/file.wav

# Helper scripts (output to assets/)
python -m scripts.generate_report_spectrograms
python -m scripts.generate_snr_distance_graph
python -m scripts.generate_augmentation_comparison
python -m scripts.visualize_mel_filters
python -m scripts.generate_sliding_window_diagram

# Data helpers
python -m scripts.count_samples           # detailed metadata stats
python -m scripts.count_samples_fast      # quick stats (no pandas)
python -m scripts.update_metadata
python -m scripts.export_misclassified
python -m scripts.test_segmentation
```

Always activate the venv first: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux).

## Key architecture

| Directory | Purpose |
|---|---|
| `src/app/` | PyQt6 dashboard UI, background worker thread, UDP audio source |
| `src/common/` | DSP: `preprocess_pcm_audio`, `extract_mel_spectrogram` |
| `src/training/` | `data_loader.py` (segmentation + augmentation), `train.py` (CNN model) |
| `scripts/` | Standalone utilities (figures, counting, misclassification export) |
| `data/raw/` | Source `.wav` files in `drone/`, `background/` subdirs |
| `data/processed/` | Extracted `.npy` features (features.npy, labels.npy, source_file_indices.npy) |
| `models/` | Trained `.keras` files (LFS-tracked) |
| `styles/` | `style.qss` — Qt stylesheet |

## DSP pipeline

- **Sample rate**: 16000 Hz (entire system, from training to UDP inference)
- **Segment**: 1.0s windows, 50% overlap (hop=0.5s)
- **STFT**: `n_fft=2048`, `hop_length=512`, Hann window (librosa default)
- **Mel bands**: 128
- **Output shape**: (128, 128) → padded/truncated to 128 time steps → reshaped to (128, 128, 1) for CNN

## CNN model

- 4 Conv2D blocks: filters 32→64→128→256, each with BN + MaxPool(2,2) + Dropout(0.25)
- 2 Dense layers: 256, 128, each with Dropout(0.5)
- Output: 1 neuron, sigmoid, binary crossentropy
- Threshold: `> 0.5` for DRONE label (ARCHITECTURE_AND_VOTING.md mentions 0.4 but code uses 0.5)
- Optimizer: Adam lr=0.001 + ReduceLROnPlateau + EarlyStopping(patience=15)

## Voting mechanism

Temporal majority voting in `threads.py` (`_register_vote`): window of 3 inference segments, threshold ≥ 2/3. Each inference runs every 0.5s. With < 3 votes, any positive wins.

## Gotchas

- All entrypoints set `os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"` and `TF_CPP_MIN_LOG_LEVEL = "2"` at module top — preserve these when adding new training/inference scripts.
- `threads.py` uses `sys.path.insert(0, ...)` to import from `src.*` — running from project root is required.
- `metadata.csv` uses semicolon (`;`) delimiter with UTF-8 BOM — pandas reads it with `sep=';', encoding='utf-8-sig'`.
- `.keras` and `.npy` files are tracked via Git LFS (`.gitattributes`). Ensure LFS is installed before committing these.
- The `train_output.txt` at root is a log artifact; the `udp_test2/` directory contains external hardware test scripts not part of the Python project.

## Reference docs

- `ARCHITECTURE_AND_VOTING.md` — voting algorithm and CNN architecture details
- `MODEL_SPECS.md` — exact DSP and model parameters
- `WINDOW_TECHNIQUES.md` — sliding window and Hanning window theory
- `DEBUG_CHECKLIST.md` — troubleshooting hardware/model loading
- `QUICK_START_DEBUG.md` — minimal debug checklist
