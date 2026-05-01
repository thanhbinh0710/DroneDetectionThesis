# Drone Detection System - Troubleshooting Guide

## Summary of Issues & Fixes Applied

### Issue 1: Model Not Loading ❌
**Problem**: Model file may not exist or cannot be loaded, causing silent fallback to simulated predictions.  
**Solution**: I added clear error messages in console output.

### Issue 2: No Visual Indication of Simulation ❌
**Problem**: You can't tell if results are from real model or simulation.  
**Solution**: Added "MODEL STATUS" card in dashboard:
- ✓ GREEN: Real model is running
- ⚠ RED: Using simulated predictions  
- ● ORANGE: Paused (waiting for data)

### Issue 3: Silent Error Handling ❌
**Problem**: Exceptions caught and swallowed without proper logging.  
**Solution**: Added detailed error messages with stack traces.

---

## Quick Start: Debug Your System

### Step 1: Run with Debug Enabled
```bash
# Windows
set DRONE_DEBUG=1 && python src/app/main.py

# Linux/Mac
DRONE_DEBUG=1 python src/app/main.py
```

**Watch console for these messages:**

✅ **Success - Model Loaded:**
```
✓ Model loaded successfully from: f:\Python_project\...\models\drone_model_v1.keras
✓ UDP audio source listening on 192.168.x.x:5555
[UDP] packets=X, bytes=Y, buffer_size=Z/31488
```

❌ **Failure - Model Not Found:**
```
✗ ERROR: Model not found at: ...
   Expected path: f:\Python_project\...\models\drone_model_v1.keras
   USING SIMULATED PREDICTIONS - no real detections are running!
```

❌ **Failure - TensorFlow Not Installed:**
```
✗ ERROR: Failed to load model: No module named 'tensorflow'
   Error type: ModuleNotFoundError
   USING SIMULATED PREDICTIONS - no real detections are running!
```

---

## Step 2: Verify Files Exist

### Check 1: Model File
```bash
# Windows - Go to project root and check:
dir models\drone_model_v1.keras

# Should show file size > 0
```

If missing:
- Train model: `python src/training/train.py`
- Or copy from backup if available

### Check 2: Audio Processing Modules
```bash
# Check processor.py exists
dir src\common\processor.py

# Check data_loader.py exists  
dir src\training\data_loader.py
```

### Check 3: Required Dependencies
```bash
# Check TensorFlow installed
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

# Check librosa installed
python -c "import librosa; print(f'Librosa version: {librosa.__version__}')"
```

If any are missing, install:
```bash
pip install tensorflow librosa numpy
```

---

## Step 3: Check UDP Connection

### Windows:
```bash
# Check if port 5555 is listening
netstat -ano | findstr :5555

# Should show your Python process
```

### Linux/Mac:
```bash
ss -an | grep 5555
```

### Verify Hardware Sending Data:
1. Run the app with `DRONE_DEBUG=1`
2. Check console for:
   ```
   [UDP] packets=5, bytes=10240, buffer_size=1024/31488
   ```
   - If packets > 0: Hardware is sending ✅
   - If packets = 0: Hardware not sending ❌

---

## Step 4: Understand Dashboard Indicators

### MODEL STATUS Card
```
✓ REAL MODEL     = Model loaded, running predictions on real audio
⚠ SIMULATED      = Model not found/failed to load, showing random values
● PAUSED         = Waiting for enough audio data (buffer not full yet)
```

### System Results
```
DRONE  (green)   = Model detected drone sound with > 50% confidence
  -    (gray)    = No drone detected (< 50% confidence)
```

### AUDIO SOURCE Card
```
UDP 192.168.x.x:5555 | 16000 Hz
```
- Should show hardware IP and port
- If shows "UDP (waiting)": Not receiving data yet

---

## Debugging Your Hardware Connection

### Scenario 1: Model Status = "⚠ SIMULATED"

**Solution:**
1. Check model file exists: `dir models\drone_model_v1.keras`
2. Run: `set DRONE_DEBUG=1 && python src/app/main.py`
3. Look for error in console:
   - "Model not found at:" → File missing, train new model
   - "Failed to load model:" → TensorFlow error, check installation
   - "No module named 'tensorflow'" → Install TensorFlow

### Scenario 2: Audio Source = "UDP (waiting)" + No Buffer Growth

**Solution:**
1. Verify hardware is sending data:
   ```bash
   # Terminal 2: Check if receiving
   python -c "
   import socket
   s = socket.socket()
   s.bind(('0.0.0.0', 5555))
   print('Listening...')
   data, addr = s.recvfrom(1024)
   print(f'Received {len(data)} bytes from {addr}')
   "
   ```
2. If no data received: Check hardware IP/port configuration
3. If data received: Model should auto-update after buffer fills (~2 seconds)

### Scenario 3: Model Running but Always Shows "-" (No Drone Detected)

**Solution:**
1. Check confidence scores in history graph
2. If all near 0.0: Audio preprocessing may be wrong
3. If all near 1.0: Model may have bias
4. Try different audio samples (loud drone sound vs background noise)
5. Check model performance with: `python src/training/train.py --evaluate`

---

## Buffer Calculation

Current settings:
- Source sample rate: 16,000 Hz
- Target sample rate: 44,100 Hz
- Mel-spectrogram time steps: 128
- Required audio: ~0.78 seconds
- **Required bytes**: 31,488 bytes (≈ 1.96 seconds at 16kHz, 16-bit)

The app waits for this much data before running the model. This is **normal**.

Time to first prediction: **~2-3 seconds** of audio.

---

## Common Error Messages & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `Model not found at: ...` | File missing | Train model: `python src/training/train.py` |
| `No module named 'tensorflow'` | TensorFlow not installed | `pip install tensorflow` |
| `No module named 'librosa'` | Librosa not installed | `pip install librosa` |
| `PAUSED` (stuck) | Buffer never fills | Check hardware connection |
| `SIMULATED` (red warning) | Model failed to load | Check console for error message |
| ValueError in reshape | Model expects different input shape | Retrain model or check audio preprocessing |

---

## Next Steps If Still Not Working

1. **Enable full debug mode** and run for 5+ seconds
2. **Share the console output** with:
   - Model loading status
   - Any ERROR messages
   - UDP packet counts
3. **Verify hardware** is actually running and sending data
4. **Check model file** isn't corrupted:
   ```bash
   python -c "import tensorflow as tf; m = tf.keras.models.load_model('models/drone_model_v1.keras'); print(m.summary())"
   ```

---

## Files Modified (Today's Fixes)

- `src/app/threads.py`: Better error messages, debug logging, simulation indicator
- `src/app/main.py`: Added MODEL STATUS indicator, source tracking
- `DEBUG_CHECKLIST.md`: This guide

