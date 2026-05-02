# QUICK CHECKLIST - Run This First!

Run your app with debug enabled, then check these items:

## 1️⃣ Check Console Output
```bash
set DRONE_DEBUG=1 && python src/app/main.py
```

In the console, do you see:

- [ ] ✓ Model loaded successfully from: ...?
- [ ] ✓ UDP audio source listening on ...?
- [ ] [UDP] packets showing numbers > 0?

**If MODEL says "✗ ERROR" → GO TO ISSUE #3 BELOW**  
**If UDP shows "packets=0" → GO TO ISSUE #2 BELOW**

---

## 2️⃣ Dashboard - MODEL STATUS Card

Open the app and look at the top cards. The "MODEL STATUS" card should show:

- [ ] ✓ REAL MODEL (GREEN) → Model is working! ✅
- [ ] ⚠ SIMULATED (RED) → Model not loaded, using fake data ❌
- [ ] ● PAUSED (ORANGE) → Waiting for audio data

**If you see SIMULATED or ERROR → GO TO ISSUE #3**  
**If you see PAUSED for > 5 seconds → GO TO ISSUE #2**

---

## 3️⃣ If Model Shows ERROR or SIMULATED

Check if model file exists:

```bash
# Windows - from project root
dir models\drone_model_v1.keras

# Should show file size (e.g., "drone_model_v1.keras 1,234,567 bytes")
```

- [ ] File exists with size > 1 MB?

**If NO:** Your model file is missing. Options:
1. Train a new model: `python src/training/train.py`
2. Download from backup/git repo

**If YES:** Check TensorFlow:
```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

- [ ] No error? (Should print version like "2.13.0")

**If ERROR:** Install TensorFlow:
```bash
pip install tensorflow
```

---

## 4️⃣ If Audio Shows PAUSED or Buffer Not Filling

Your hardware is not sending data. Check:

1. Hardware IP/Port Configuration
   - ESP32 configured to send to port **5555**?
   - Connected to same network as PC?

2. Verify PC is listening:
   ```bash
   netstat -ano | findstr :5555
   ```
   - Should show your Python app listening

3. Test UDP manually (Terminal 2):
   ```bash
   python -c "
   import socket
   s = socket.socket()
   s.bind(('0.0.0.0', 5555))
   print('Waiting for data...')
   data, addr = s.recvfrom(1024)
   print(f'Got {len(data)} bytes from {addr}')
   "
   ```
   
   Then send data from hardware and check if it's received.

---

## 5️⃣ If Everything Loads but No Results Change

- [ ] Confidence history graph shows smooth line?
- [ ] Status changes between "DRONE" and "-"?

**If graph is flat/always 0:** Model predictions aren't working
- Check audio format: should be 16kHz, 16-bit mono PCM
- Retrain model if audio format doesn't match

**If graph is flat/always high:** Model is confident but wrong
- Check with different audio samples
- Model may need retraining with your hardware's audio characteristics

---

## ✅ SUCCESS INDICATORS

When everything is working, you should see:

1. **Console Output:**
   ```
   ✓ Model loaded successfully from: ...
   ✓ UDP audio source listening on 192.168.x.x:5555
   [UDP] packets=10, bytes=20480, buffer_size=25600/31488
   ```

2. **Dashboard:**
   - MODEL STATUS: `✓ REAL MODEL` (GREEN)
   - System Results: Alternates between `DRONE` and `-`
   - Confidence: Changes from ~0.1 to ~0.9
   - History Graph: Shows smooth variations

3. **Behavior:**
   - Results update every ~0.5 seconds
   - Drone sound → status shows `DRONE` (green)
   - Background noise → status shows `-` (gray)
   - Detection count increases when drone detected

---

## 🆘 Still Not Working?

1. **Record console output** (copy-paste all text)
2. **Take screenshot** of dashboard
3. **Share what you're seeing:**
   - Does model load? (check console)
   - Does UDP receive data? (check console for "packets=")
   - What does MODEL STATUS card show?

Then I can help diagnose the exact issue!
