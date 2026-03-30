# Drone Detection Models

This directory contains trained machine learning models for drone detection.

## Expected Files

- `drone_model_v1.keras` - Trained CNN model for drone detection from mel-spectrograms (native Keras format)
- `scaler.pkl` - Feature scaler (StandardScaler or MinMaxScaler) for normalization

## Model Training

To train a new model, run:

```bash
python -m src.training.train
```

Models will be automatically saved to this directory after training.

## Model Usage

The models can be loaded in the application using:

```python
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model('models/drone_model_v1.keras')

# Load scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

## Model Versions

| Version | Date | Accuracy | Notes         |
| ------- | ---- | -------- | ------------- |
| v1      | TBD  | TBD      | Initial model |
