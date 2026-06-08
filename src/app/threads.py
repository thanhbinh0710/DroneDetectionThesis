# FILE: src/app/threads.py
import time
import os
import gc
import random
from collections import deque
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# Limit TF threads for single-sample inference predictability
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

# Configure which model to load here
# Example: "drone_model_20260507_143000.keras"
MODEL_FILENAME = "drone_model_20260509_111413.keras"


class DataWorker(QThread):
    # Signal: emit prediction data for dashboard update
    data_signal = pyqtSignal(dict)  # Changed to dict to pass more info

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.counter = 0
        self.model = None
        self.audio_source = None
        self.audio_buffer = bytearray()
        self.debug = os.environ.get("DRONE_DEBUG", "0") == "1"
        self._last_debug_ts = 0.0
        if self.debug:
            print("Debug enabled: DRONE_DEBUG=1")

        self.source_sr = 16000
        self.target_sr = 16000
        self.n_fft = 2048
        self.hop_length = 512
        self.target_mel_length = 128
        self.required_target_samples = (
            self.target_mel_length - 1
        ) * self.hop_length + self.n_fft
        self.required_input_samples = int(
            np.ceil(self.required_target_samples * self.source_sr / self.target_sr)
        )
        self.required_input_bytes = self.required_input_samples * 2

        # Sliding-window majority voting over the latest 3 inference segments
        self.voting_window_size = 3
        self.inference_interval_seconds = 0.5
        self.voting_history = deque(maxlen=self.voting_window_size)

        # Track whether model was ever successfully loaded
        self.model_loaded = False
        # Load model and setup
        self._load_model()
        self._init_audio_source()

    def _load_model(self):
        """Load the trained CNN model"""
        try:
            import tensorflow as tf

            # Get model path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, "../.."))
            model_path = os.path.join(project_root, "models", MODEL_FILENAME)

            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                self.model_loaded = True
                print(f"Model loaded successfully from: {model_path}")
            else:
                print(f"ERROR: Model not found at: {model_path}")
                print(f"   Expected path: {model_path}")
                print(
                    "   USING SIMULATED PREDICTIONS - no real detections are running!"
                )
                self.model = None
                self.model_loaded = False
        except Exception as e:
            print(f"✗ ERROR: Failed to load model: {e}")
            print(f"   Error type: {type(e).__name__}")
            print("   USING SIMULATED PREDICTIONS - no real detections are running!")
            self.model = None
            self.model_loaded = False

    def _init_audio_source(self):
        """Initialize UDP audio source"""
        try:
            from src.app.hardware import UdpAudioSource

            self.audio_source = UdpAudioSource(
                port=5555,
                chunk_frames=1024,
                sample_rate=self.source_sr,
                channels=1,
                sample_width_bytes=2,
                timeout_s=0.05,
            )
            self.audio_source.start()
            print(
                f"UDP audio source listening on {self.audio_source.get_device_info()}"
            )
        except Exception as e:
            print(f"Warning: Error initializing UDP source: {e}")
            self.audio_source = None

    def _ensure_audio_source(self):
        """Ensure audio source exists; does NOT consume any data."""
        if self.audio_source is None:
            self._init_audio_source()

    def _read_and_buffer(self):
        """Read chunks from UDP, trim buffer, return (device_info, sample_rate) or None."""
        if self.audio_source is None:
            return None
        device_info = self.audio_source.get_device_info()
        sample_rate = self.audio_source.sample_rate
        old_len = len(self.audio_buffer)
        chunks = self.audio_source.read_available(max_reads=50)
        for chunk in chunks:
            self.audio_buffer.extend(chunk)
        new_len = len(self.audio_buffer)
        if self.debug:
            now = time.time()
            if now - self._last_debug_ts >= 1.0:
                total_bytes = sum(len(c) for c in chunks)
                print(
                    f"[UDP] packets={len(chunks)}, bytes={total_bytes}, "
                    f"buffer={new_len}/{self.required_input_bytes}"
                )
                self._last_debug_ts = now
        # Detect stalled buffer: no data progress for >= 10 seconds (20 cycles)
        if len(chunks) == 0 and new_len == old_len and new_len < self.required_input_bytes:
            self._stall_count = getattr(self, '_stall_count', 0) + 1
        else:
            self._stall_count = 0
        if self._stall_count >= 20:
            print(f"[UDP] Buffer stalled ({self._stall_count} cycles), recreating socket...")
            try:
                self.audio_source.stop()
            except Exception:
                pass
            self.audio_source = None
            self.audio_buffer.clear()
            self._init_audio_source()
            return None
        # LUÔN trim buffer TRƯỚC early-return — tránh memory leak
        if new_len > self.required_input_bytes * 2:
            self.audio_buffer = self.audio_buffer[-self.required_input_bytes:]
        if len(self.audio_buffer) < self.required_input_bytes:
            if self.debug:
                now = time.time()
                if now - self._last_debug_ts >= 1.0:
                    print(f"[BUFFER] have {len(self.audio_buffer)}/{self.required_input_bytes} bytes")
                    self._last_debug_ts = now
            return None
        return device_info, sample_rate

    def _preprocess_to_mel(self, window_bytes):
        """Convert raw PCM bytes to mel-spectrogram, or None on failure."""
        from src.common.processor import preprocess_pcm_audio, extract_mel_spectrogram
        from src.training.data_loader import pad_or_truncate_spectrogram

        audio = preprocess_pcm_audio(
            window_bytes,
            input_sr=self.source_sr,
            target_sr=self.target_sr,
            trim_silence=False,
        )
        if audio.size == 0:
            return None
        mel_spec = extract_mel_spectrogram(audio, sr=self.target_sr)
        return pad_or_truncate_spectrogram(mel_spec, target_length=self.target_mel_length)

    def _get_prediction(self):
        """Orchestrate: read audio → preprocess → inference → return result dict."""
        self._ensure_audio_source()

        result = self._read_and_buffer()
        if result is None:
            device_info = getattr(self.audio_source, 'get_device_info', lambda: "N/A")()
            return self._paused_prediction(device_info, self.source_sr)
        device_info, sample_rate = result

        if self.model is None:
            if self.debug:
                print("[MODEL] No model — SIMULATED predictions")
            return self._simulate_prediction(device_info, sample_rate)

        try:
            window_bytes = self.audio_buffer[-self.required_input_bytes:]
            mel_spec = self._preprocess_to_mel(window_bytes)
            if mel_spec is None:
                if self.debug:
                    print("[AUDIO] Preprocessed audio is empty!")
                return self._paused_prediction(device_info, sample_rate)

            X = mel_spec.reshape(1, self.target_mel_length, self.target_mel_length, 1)
            pred = self.model.predict_on_batch(X).numpy()[0, 0]

            return {
                "confidence": float(pred),
                "status": "DRONE" if pred > 0.5 else "-",
                "source": "real_model",
                "file": "UDP",
                "device_info": device_info,
                "sample_rate": sample_rate,
                "paused": False,
            }
        except Exception as e:
            print(f"✗ ERROR in prediction: {e}")
            import traceback
            traceback.print_exc()
            return self._paused_prediction(device_info, sample_rate)

    def _register_vote(self, prediction):
        """Store a prediction and aggregate the latest 3 inference segments."""
        raw_confidence = float(prediction["confidence"])
        raw_status = prediction["status"]
        is_drone = raw_status == "DRONE"

        self.voting_history.append(
            {
                "is_drone": is_drone,
                "confidence": raw_confidence,
                "source": prediction.get("source", "unknown"),
                "file": prediction.get("file", "UDP"),
                "device_info": prediction.get("device_info"),
                "sample_rate": prediction.get("sample_rate"),
                "paused": prediction.get("paused", False),
            }
        )

        window_votes = list(self.voting_history)
        total_votes = len(window_votes)
        drone_votes = sum(1 for item in window_votes if item["is_drone"])
        drone_ratio = drone_votes / total_votes if total_votes else 0.0

        # Fallback for short clips: with fewer than 3 segments, any positive vote wins.
        if total_votes < self.voting_window_size:
            aggregated_is_drone = drone_votes >= 1
        else:
            # Standard rule: 2 of 3 segments must vote DRONE.
            aggregated_is_drone = drone_votes >= 2

        aggregated_status = "DRONE" if aggregated_is_drone else "-"
        aggregated_confidence = drone_ratio

        return {
            "confidence": float(aggregated_confidence),
            "status": aggregated_status,
            "source": prediction.get("source", "unknown"),
            "file": prediction.get("file", "UDP"),
            "device_info": prediction.get("device_info"),
            "sample_rate": prediction.get("sample_rate"),
            "paused": prediction.get("paused", False),
            "voting_window_size": self.voting_window_size,
            "vote_count": total_votes,
            "drone_votes": drone_votes,
            "drone_ratio": float(drone_ratio),
            "raw_confidence": raw_confidence,
            "raw_status": raw_status,
        }

    def _paused_prediction(self, device_info, sample_rate):
        return {
            "confidence": 0.0,
            "status": "PAUSED",
            "source": "udp",
            "file": "UDP",
            "device_info": device_info,
            "sample_rate": sample_rate,
            "paused": True,
        }

    def _simulate_prediction(self, device_info=None, sample_rate=None):
        """Simulate prediction when model is not available"""
        confidence = random.uniform(0.60, 0.95)
        return {
            "confidence": confidence,
            "status": "DRONE" if confidence > 0.65 else "-",
            "source": "simulated",
            "file": "N/A",
            "device_info": device_info or "N/A",
            "sample_rate": sample_rate or self.source_sr,
            "paused": False,
        }

    def run(self):
        self.is_running = True
        cleanup_counter = 0
        while self.is_running:
            try:
                # Get prediction (real or simulated)
                prediction = self._get_prediction()

                if not prediction.get("paused", False):
                    prediction = self._register_vote(prediction)

                # Emit signal with prediction data
                self.data_signal.emit(prediction)
                time.sleep(self.inference_interval_seconds)

                self.counter += 1

                # Periodic GC cleanup every ~5 minutes (600 * 0.5s)
                cleanup_counter += 1
                if cleanup_counter >= 600:
                    gc.collect()
                    cleanup_counter = 0
            except Exception as e:
                print(f"✗ CRITICAL ERROR in worker thread: {e}")
                import traceback

                traceback.print_exc()
                # Continue running even if there's an error
                time.sleep(1.0)

    def stop(self):
        """Stop the worker thread gracefully"""
        self.is_running = False
        if self.audio_source:
            try:
                self.audio_source.stop()
            except Exception as e:
                print(f"Error stopping audio source: {e}")
        self.wait()  # Wait for thread to finish

    