# FILE: src/app/threads.py
import time
import os
import glob
import random
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class DataWorker(QThread):
    # Signal: emit prediction data for dashboard update
    data_signal = pyqtSignal(dict)  # Changed to dict to pass more info

    def __init__(self):
        super().__init__()
        self.is_running = False
        self.counter = 0
        self.model = None
        self.audio_files = []
        self.current_audio_idx = 0
        
        # Load model and setup
        self._load_model()
        self._load_audio_files()

    def _load_model(self):
        """Load the trained CNN model"""
        try:
            import tensorflow as tf
            
            # Get model path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, '../..'))
            model_path = os.path.join(project_root, 'models', 'drone_model_v1.keras')
            
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"✓ Model loaded from: {model_path}")
            else:
                print(f"⚠️  Model not found at: {model_path}")
                print("   Using simulated predictions instead")
                self.model = None
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("   Using simulated predictions instead")
            self.model = None

    def _load_audio_files(self):
        """Load list of audio files for testing"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, '../..'))
            drone_dir = os.path.join(project_root, 'data', 'raw', 'drone')
            
            # Get all DRONE audio files from drone subdirectory
            self.audio_files = sorted(glob.glob(os.path.join(drone_dir, 'DRONE_*.wav')))
            
            if self.audio_files:
                print(f"✓ Found {len(self.audio_files)} audio files for testing")
            else:
                print(f"⚠️  No audio files found in {drone_dir}")
        except Exception as e:
            print(f"⚠️  Error loading audio files: {e}")
            self.audio_files = []

    def _get_prediction(self):
        """Get prediction from model or simulate"""
        if self.model and self.audio_files:
            try:
                from src.common.processor import preprocess_audio, extract_mel_spectrogram
                from src.training.data_loader import pad_or_truncate_spectrogram
                
                # Cycle through audio files
                audio_file = self.audio_files[self.current_audio_idx]
                self.current_audio_idx = (self.current_audio_idx + 1) % len(self.audio_files)
                
                # Process audio
                audio = preprocess_audio(audio_file)
                mel_spec = extract_mel_spectrogram(audio)
                mel_spec = pad_or_truncate_spectrogram(mel_spec, target_length=128)
                
                # Reshape for model input (1, 128, 128, 1)
                X = mel_spec.reshape(1, 128, 128, 1)
                
                # Get prediction
                prediction = self.model.predict(X, verbose=0)[0][0]
                
                return {
                    'confidence': float(prediction),
                    'status': 'DRONE' if prediction > 0.5 else '-',
                    'source': 'model',
                    'file': os.path.basename(audio_file)
                }
            except Exception as e:
                print(f"⚠️  Prediction error: {e}")
                # Fallback to simulation
                return self._simulate_prediction()
        else:
            return self._simulate_prediction()

    def _simulate_prediction(self):
        """Simulate prediction when model is not available"""
        confidence = random.uniform(0.60, 0.95)
        return {
            'confidence': confidence,
            'status': 'DRONE' if confidence > 0.65 else '-',
            'source': 'simulated',
            'file': 'N/A'
        }

    def run(self):
        self.is_running = True
        while self.is_running:
            # Get prediction (real or simulated)
            prediction = self._get_prediction()
            
            # Emit signal with prediction data
            self.data_signal.emit(prediction)
            time.sleep(2.0)  # Update every 2 seconds to allow audio processing
            
            self.counter += 1

    def stop(self):
        self.is_running = False
