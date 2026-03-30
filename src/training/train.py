# src/training/train.py
"""
Audio-based Drone Detection Model Training Script
Train CNN models from preprocessed mel-spectrograms for drone sound classification
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.common.processor import preprocess_audio, extract_mel_spectrogram


def load_processed_data(processed_dir):
    """
    Load preprocessed mel-spectrogram features from data/processed/
    
    Args:
        processed_dir: Path to processed data directory
        
    Returns:
        X: Feature array (mel-spectrograms) - shape: (n_samples, 128, 128)
        y: Labels (1=DRONE, 0=NOT_DRONE)
    """
    features_path = os.path.join(processed_dir, 'features.npy')
    labels_path = os.path.join(processed_dir, 'labels.npy')
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Processed data not found in {processed_dir}\n"
            "Please run: python -m src.training.data_loader"
        )
    
    X = np.load(features_path)
    y = np.load(labels_path)
    
    print(f"✓ Loaded processed data:")
    print(f"  - Features: {X.shape}")
    print(f"  - Labels: {y.shape}")
    
    return X, y


def build_model(input_shape=(128, 128, 1)):
    """
    Build CNN model for audio-based drone detection
    
    Args:
        input_shape: Shape of input mel-spectrogram (mel_bands, time_steps, channels)
        
    Returns:
        model: Compiled Keras model
        
    Architecture:
    - Conv2D layers: Extract frequency and temporal patterns from mel-spectrograms
    - BatchNormalization: Stabilize training
    - MaxPooling: Dimensionality reduction
    - Dropout: Prevent overfitting
    - Dense layers: Classification
    - Output: Binary classification with sigmoid activation
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Block 1: 32 filters
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.25),
        
        # Block 2: 64 filters
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.25),
        
        # Block 3: 128 filters
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(0.25),
        
        # Block 4: 256 filters
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name='pool4'),
        layers.Dropout(0.25),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu', name='dense1'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', name='dense2'),
        layers.Dropout(0.5),
        
        # Output layer: Binary classification (DRONE vs NOT_DRONE)
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, model_save_path, epochs=50, batch_size=16):
    """
    Train the drone detection model
    
    Args:
        model: Compiled CNN model
        X_train: Training mel-spectrogram features
        y_train: Training labels (1=DRONE, 0=NOT_DRONE)
        X_val: Validation features
        y_val: Validation labels
        model_save_path: Path to save best model
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        history: Training history object
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Callbacks
    callbacks = [
        # Save best model based on validation loss
        ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        
        # Early stopping if validation loss doesn't improve
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print("="*60)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    
    return history


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics
    
    Args:
        history: Training history from model.fit()
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
    axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
    axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Training plots saved to: {save_path}")
    
    plt.show()


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        results: Dictionary of evaluation metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Evaluate
    results = model.evaluate(X_test, y_test, verbose=0)
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\nTest Metrics:")
    print(f"  Loss: {results[0]:.4f}")
    print(f"  Accuracy: {results[1]:.4f}")
    print(f"  Precision: {results[2]:.4f}")
    print(f"  Recall: {results[3]:.4f}")
    
    print("\nClassification Report:")
    # Use labels parameter to handle cases with only one class present
    print(classification_report(y_test, y_pred, target_names=['NOT_DRONE', 'DRONE'], labels=[0, 1], zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(cm)
    print("(Rows: True labels, Columns: Predicted labels)")
    print("[[True Negatives, False Positives],")
    print(" [False Negatives, True Positives]]")
    
    return {
        'loss': results[0],
        'accuracy': results[1],
        'precision': results[2],
        'recall': results[3]
    }


def main():
    """Main training pipeline for audio-based drone detection"""
    print("=" * 60)
    print("AUDIO-BASED DRONE DETECTION - MODEL TRAINING")
    print("=" * 60)
    print("\nSystem: Acoustic drone detection using mel-spectrogram features")
    print("Method: CNN classification (DRONE vs NOT_DRONE)\n")
    
    # Xác định đường dẫn tuyệt đối
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    # Paths
    processed_dir = os.path.join(project_root, "data", "processed")
    model_save_path = os.path.join(project_root, "models", "drone_model_v1.keras")
    plot_save_path = os.path.join(project_root, "models", "training_history.png")
    
    print(f"📂 Project root: {project_root}")
    print(f"📂 Processed data: {processed_dir}")
    print(f"💾 Model save path: {model_save_path}\n")
    
    # 1. Load preprocessed data
    print("STEP 1: Loading preprocessed data...")
    try:
        X, y = load_processed_data(processed_dir)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return
    
    # 2. Reshape data for CNN (add channel dimension)
    print("\nSTEP 2: Preparing data for CNN...")
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)  # (n_samples, 128, 128, 1)
    print(f"✓ Reshaped features: {X.shape}")
    
    # 3. Split data into train/validation/test sets (70/15/15)
    print("\nSTEP 3: Splitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Validation set: {X_val.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # 4. Build model
    print("\nSTEP 4: Building CNN model...")
    model = build_model(input_shape=(128, 128, 1))
    print("\nModel Architecture:")
    model.summary()
    
    print(f"\n📊 Total parameters: {model.count_params():,}")
    
    # 5. Train model
    print("\nSTEP 5: Training model...")
    history = train_model(
        model, 
        X_train, y_train, 
        X_val, y_val,
        model_save_path,
        epochs=50,
        batch_size=16
    )
    
    # 6. Plot training history
    print("\nSTEP 6: Plotting training history...")
    plot_training_history(history, save_path=plot_save_path)
    
    # 7. Evaluate on test set
    print("\nSTEP 7: Evaluating model on test set...")
    test_results = evaluate_model(model, X_test, y_test)
    
    # 8. Save final summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"✓ Model saved to: {model_save_path}")
    print(f"✓ Training plots saved to: {plot_save_path}")
    print(f"\nFinal Test Results:")
    print(f"  - Accuracy: {test_results['accuracy']:.4f}")
    print(f"  - Precision: {test_results['precision']:.4f}")
    print(f"  - Recall: {test_results['recall']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
