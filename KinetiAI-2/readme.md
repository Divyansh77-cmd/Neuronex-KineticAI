# KinetiAI — Neuromorphic Rehab Monitor
### Open Neuromorphism Track · Hackathon Submission

> **Spiking Neural Networks · Event-Driven Intelligence · Post-Operative Recovery**

---

## What is KinetiAI?

KinetiAI is a browser-based, AI-powered post-surgery rehabilitation monitoring platform that uses computer vision (MediaPipe pose estimation) and an optional M5Stick wearable IMU sensor to track knee flexion angles and repetition counts in real time during physiotherapy exercises.

For this hackathon, KinetiAI has been extended with a **neuromorphic Leaky Integrate-and-Fire (LIF) Spiking Neural Network engine** that replaces the conventional threshold-based hysteresis repetition counter with a biologically-inspired, event-driven spike classifier.

---

## Neuromorphic Component: The LIF SNN Engine

### Problem with the old system
The original rep counter used hardcoded angle thresholds:
```
if angle < 110° → mark as "flexed"
if angle > 160° → count one rep
```
This detects static values, not temporal movement patterns. It cannot detect compensation movements (e.g., patient tilting their hip to fake a deeper bend) and fires identically regardless of the speed or quality of the motion.

### The SNN solution
We replaced this with a **Leaky Integrate-and-Fire (LIF) neuron** — the foundational computational unit of Spiking Neural Networks. The LIF model mimics biological neurons:

```
V[t] = V[t-1] × leak + I[t]

If V[t] ≥ threshold → FIRE SPIKE, reset V = 0
```

Where:
- `V` = membrane potential (accumulated "charge")  
- `leak` = 0.92 (membrane decay constant — models ion channel leakage)  
- `I[t]` = rate-encoded input current derived from the knee angle stream  
- `threshold` = 1.0 (firing threshold)  

### Rate Coding
The continuous knee angle stream is converted to a spike-input current using **population rate coding** — the same mechanism biological mechanoreceptor neurons use to encode joint angle:

```javascript
// Deep flexion (small angle) → high current → drives V up fast → spikes
// Full extension (large angle) → low current → V decays via leak → silence
I = ((180 - angle) / (180 - 60))²
```

The squaring nonlinearity sharpens the deep-flexion response, mimicking the nonlinear response curves of biological stretch receptors.

### SNN State Machine (rep counting)
A 4-state event-driven machine replaces the old if/else:

```
IDLE → ENCODING → FLEXION (spike in bent zone) → EXTENSION (V decays) → +1 REP → ENCODING
```

This detects the **temporal pattern** of a correct rep, not just two static angle snapshots.

### Spike Oscilloscope
A real-time canvas overlay below the camera feed renders:
- The LIF membrane potential `Vₘ` as a live waveform (purple)
- Spike fire events as vertical red markers
- The firing threshold as a dashed red line

---

## Files

| File | Description |
|------|-------------|
| `index.html` | Main patient interface — camera tracking + SNN telemetry panel + spike oscilloscope |
| `app2.js` | Full application logic — LIF neuron, rate coding, SNN state machine, BLE/Serial sensor, Gemini voice AI |
| `style2.css` | Styling including SNN telemetry block and spike canvas |
| `doctor.html` | Clinician portal — session analytics + SNN rep/spike count columns |

---

## How to run

1. Clone this repo
2. Serve via a local HTTPS server (required for camera + BLE/Serial access):
   ```bash
   npx serve .
   # or
   python3 -m http.server 8080
   ```
3. Open `https://localhost:PORT` in Chrome or Edge (not Safari — BLE/Serial not supported)
4. Enter your Gemini API key when prompted (stored in localStorage, never sent anywhere)
5. Stand sideways to the camera and begin rehab exercises

---

## Neuromorphic Alignment

| Theme | How KinetiAI addresses it |
|-------|--------------------------|
| **Spiking Neural Networks** | LIF neuron processes every camera frame; fires discrete spike events on deep flexion detection |
| **Event-Driven Intelligence** | Computation (spike + rep count) only occurs when angle change produces sufficient input current — silent during rest |
| **Brain-Inspired AI** | Rate coding, membrane potential, leak constant, refractory period — all direct analogues of biological neuron dynamics |
| **Neuromorphic for wearables** | Event-driven processing principle directly maps to low-power neuromorphic chips (Intel Loihi, BrainScaleS) for the M5Stick sensor path |

---

## Future Work

- Replace browser-JS LIF with Python/SpikingJelly SNN trained on labelled angle datasets to classify correct vs. compensation movements
- Integrate KinetiRAG (DoctorRAG-adapted RAG pipeline) for grounded, clinician-reviewable AI recommendations
- Deploy LIF inference on neuromorphic hardware co-located with the M5Stick sensor for ultra-low-power edge processing