<p align="center">
  <img src="https://img.shields.io/badge/Platform-Desktop%20%7C%20Mobile-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ML-TensorFlow%20%7C%20Keras-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Mobile-React%20Native%20%7C%20Expo%20SDK%2054-purple?style=for-the-badge&logo=react" />
  <img src="https://img.shields.io/badge/Inference-100%25%20On--Device-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

# 🤟 SignBridge — Real-Time ASL Translation System

> **A dual-platform (Desktop + Mobile) real-time American Sign Language translator powered by on-device edge AI. Translates both fingerspelling (A–Z) and dynamic phrase-level ASL into text and speech — entirely offline, with zero cloud dependency.**

SignBridge bridges the communication gap between the Deaf/hard-of-hearing community and the general public by running two custom neural networks on consumer hardware — no internet, no servers, no specialized gloves.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Demo](#-demo)
- [System Architecture](#-system-architecture)
- [ML Models](#-ml-models)
- [The Hybrid Router](#-the-hybrid-router)
- [Datasets](#-datasets)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [How It Works](#-how-it-works)
- [Novel Contributions](#-novel-contributions)
- [Future Roadmap](#-future-roadmap)
- [Team](#-team)
- [References](#-references)
- [License](#-license)

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔤 **Fingerspelling (A–Z)** | Static alphabet recognition with 28 classes (A–Z + `del` + `space`) |
| 🗣️ **Dynamic Phrases** | Recognizes 21 common ASL word-level signs (hello, thankyou, help, etc.) |
| 🔀 **Automatic Mode Switching** | Novel Momentum-Based Hybrid Router seamlessly detects spelling vs. phrase signing |
| 📱 **Dual Platform** | Desktop (Python/OpenCV) + Mobile (React Native/Expo) |
| 🔒 **100% Offline & Private** | All inference runs on-device — zero cloud calls, zero data leaves the phone |
| 🗣️ **Text-to-Speech** | Vocalizes translated sentences out loud for two-way conversation |
| 🦴 **Skeleton Overlay** | Real-time 21-joint hand skeleton visualization over camera feed |
| ⚡ **Real-Time** | 30 FPS on desktop, 15–24 FPS on mobile |

---

## 🎬 Demo

### Desktop Client (Python/OpenCV)
```
Camera feed with live skeleton tracking, mode indicator,
word construction panel, and sentence output with TTS.
Controls: C (clear) | P (speak) | T (save transcript) | Q (quit)
```

### Mobile Client (React Native)
```
Full-screen camera with SVG skeleton overlay,
glassmorphism UI panels, confidence badges,
FPS counter, and touch-based controls.
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SignBridge System                         │
├─────────────────────────┬───────────────────────────────────────┤
│     DESKTOP CLIENT      │          MOBILE CLIENT                │
│  (Python / OpenCV)      │  (React Native / Expo SDK 54)         │
├─────────────────────────┼───────────────────────────────────────┤
│                         │                                       │
│  ┌─────────────┐        │  ┌──────────────────┐                 │
│  │  USB Webcam  │        │  │  cameraWithTensors│                │
│  │  1280×720    │        │  │  (TensorCamera)   │                │
│  └──────┬──────┘        │  └────────┬─────────┘                 │
│         ▼               │           ▼                           │
│  ┌─────────────┐        │  ┌──────────────────┐                 │
│  │  MediaPipe   │        │  │  hand-pose-detect │                │
│  │  HandLandmark│        │  │  (MediaPipe lite) │                │
│  └──────┬──────┘        │  └────────┬─────────┘                 │
│         ▼               │           ▼                           │
│  ┌──────────────────────┴───────────────────────┐               │
│  │         HYBRID ROUTER (State Machine)         │               │
│  │  ┌────────────────────────────────────────┐   │               │
│  │  │  Wrist Velocity Calculator             │   │               │
│  │  │  velocity = euclidean_dist / frames    │   │               │
│  │  └──────────────┬─────────────────────────┘   │               │
│  │                 ▼                             │               │
│  │  velocity ≤ 0.025     velocity > 0.025        │               │
│  │       ▼                      ▼                │               │
│  │  ┌──────────┐        ┌────────────┐           │               │
│  │  │ SPELLING │        │ RECORDING  │           │               │
│  │  │  MODE    │        │  PHRASE    │           │               │
│  │  └────┬─────┘        └──────┬─────┘           │               │
│  │       ▼                     ▼                 │               │
│  │  ┌──────────┐        ┌────────────┐           │               │
│  │  │ Alphabet │        │  Dynamic   │           │               │
│  │  │ Model    │        │  Model     │           │               │
│  │  │ (Dense)  │        │(Conv1D+LSTM│           │               │
│  │  └────┬─────┘        └──────┬─────┘           │               │
│  └───────┴──────────┬──────────┘                 │               │
│                     ▼                                           │
│           ┌──────────────────┐                                  │
│           │  Sentence Buffer │                                  │
│           │  + TTS Engine    │                                  │
│           └──────────────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
```

### ML Training Pipeline

```
┌──────────────┐     ┌───────────────┐     ┌─────────────┐     ┌───────────────┐
│ Kaggle       │     │ MediaPipe     │     │ TensorFlow  │     │ Model Export  │
│ Datasets     │────▶│ Landmark      │────▶│ Keras       │────▶│ .h5 → .tflite │
│ (Images +    │     │ Extraction    │     │ Training    │     │ .h5 → TFJS    │
│  Parquet)    │     │ (63 features) │     │ (GPU: T4)   │     │ (model.json)  │
└──────────────┘     └───────────────┘     └─────────────┘     └───────────────┘
```

---

## 🤖 ML Models

### Model 1: Static Alphabet Classifier (Dense NN)

| Attribute | Value |
|---|---|
| **Architecture** | Dense Neural Network (Sequential) |
| **Input** | `(1, 63)` — 21 landmarks × 3 coords |
| **Output** | `(1, 28)` — 28 classes (A–Z + del + space) |
| **Parameters** | **20,220** |
| **Training** | From scratch (no transfer learning) |
| **Optimizer** | Adam (LR=0.001, ReduceLROnPlateau) |
| **Loss** | Sparse Categorical Cross-Entropy |
| **Model Size** | 289.4 KB (H5) / 23.8 KB (TFLite) |

```
Input(63) → Dense(128,ReLU) → BN → Dropout(0.3)
          → Dense(64,ReLU)  → BN → Dropout(0.2)
          → Dense(32,ReLU)  → Dropout(0.1)
          → Dense(28,Softmax)
```

### Model 2: Dynamic Phrase Classifier (Conv1D + BiLSTM)

| Attribute | Value |
|---|---|
| **Architecture** | Conv1D → MaxPool → BiLSTM → BiLSTM → Dense → Softmax |
| **Input** | `(1, 30, 63)` — 30 frames × 63 features |
| **Output** | `(1, 21)` — 21 phrase classes |
| **Parameters** | **~402,837** |
| **Training** | From scratch (no transfer learning) |
| **Optimizer** | Adam (LR=0.001, ReduceLROnPlateau) |
| **Loss** | Sparse Categorical Cross-Entropy |
| **Model Size** | 4.72 MB (H5) / 502.8 KB (TFLite) |

```
Input(30,63) → Conv1D(64,k=3,ReLU) → MaxPool(2) → BN
             → BiLSTM(128,return_seq) → BN
             → BiLSTM(64) → BN
             → Dense(128,ReLU) → BN → Dropout(0.4)
             → Dense(64,ReLU)  → BN → Dropout(0.4)
             → Dense(21,Softmax)
```

**21 Supported Dynamic Phrases:**
```
hello · yes · no · thankyou · please · fine · who · where · why
water · food · sleep · hungry · thirsty · sick · go · wait · drink · read · talk · listen
```

---

## 🔀 The Hybrid Router

The **Momentum-Based Hybrid Router** is a finite state machine that automatically detects whether the user is fingerspelling (hand still) or signing a dynamic phrase (hand moving):

```
WAITING ──(hand detected)──→ SPELLING ──(velocity > 0.025)──→ RECORDING PHRASE
   ↑                              ↑                                    │
   │                              │                       (still > patience)
   └──(no hand)──────────────────┘←──────────────────── PROCESSING PHRASE
```

**Core Logic — Euclidean Wrist Velocity:**
```python
velocity = sqrt((wrist_now.x - wrist_old.x)² +
                (wrist_now.y - wrist_old.y)² +
                (wrist_now.z - wrist_old.z)²) / FRAMES_FOR_VELOCITY
```

| Parameter | Desktop | Mobile | Purpose |
|---|---|---|---|
| `WRIST_VELOCITY_THRESHOLD` | 0.025 | 0.025 | Still vs. moving cutoff |
| `FRAMES_FOR_VELOCITY` | 5 | 5 | Velocity window size |
| `MOTION_PATIENCE` | 12 | 5 | Wait frames before prediction |
| `CONSISTENT_FRAMES_THRESHOLD` | 15 | 8 | Frames for letter confirmation |
| `CONFIDENCE_THRESHOLD` | 0.70 / 0.75 | 0.70 / 0.75 | Min confidence (static/dynamic) |

---

## 📊 Datasets

### Dataset 1: ASL Alphabet (Static Signs)

| Attribute | Detail |
|---|---|
| **Source** | [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) |
| **Type** | 200×200 RGB images |
| **Total Images** | 87,000 |
| **Classes** | 28 (A–Z + del + space) |
| **Extracted Samples** | **63,581** (via MediaPipe landmark extraction) |
| **Features** | 63 floats per sample (21 landmarks × 3 coords) |
| **Split** | 80/20 train/test (stratified) |

### Dataset 2: ASL Signs (Dynamic Phrases)

| Attribute | Detail |
|---|---|
| **Source** | [Kaggle ASL Signs Competition](https://www.kaggle.com/competitions/asl-signs/data) |
| **Type** | Pre-extracted landmarks in Parquet format |
| **Total Available** | 250 signs (~94,000 sequences) |
| **Signs Used** | 21 target phrases |
| **Samples/Sign** | 250 (balanced) |
| **Total Sequences** | ~5,250 |
| **Sequence Length** | 30 frames (padded/resampled) |
| **Features/Frame** | 63 (filtered from 543 full-body landmarks) |

### Key Preprocessing: Wrist-Relative Normalization

All landmarks are normalized relative to the wrist position, solving the "Floating Hand" problem:
```
Original:   wrist=(0.4, 0.6, 0.1), index_tip=(0.5, 0.3, 0.15)
Normalized: wrist=(0.0, 0.0, 0.0), index_tip=(0.1, -0.3, 0.05)
```
This ensures **translation and scale invariance** — hand position/distance from camera doesn't affect predictions.

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **Training Platform** | Kaggle Notebooks (NVIDIA T4 GPU) | — |
| **ML Framework** | TensorFlow / Keras | 2.x |
| **Hand Detection** | Google MediaPipe | latest |
| **Desktop Runtime** | Python + OpenCV | 3.10+ |
| **Mobile Framework** | React Native + Expo | RN 0.81.5 / Expo SDK 54 |
| **Mobile ML** | TensorFlow.js | 4.22.0 |
| **Mobile Hand Detect** | @tensorflow-models/hand-pose-detection | 2.0.1 |
| **Skeleton Rendering** | react-native-svg (mobile) / OpenCV (desktop) | 15.12.1 |
| **Data Processing** | NumPy, scikit-learn, PyArrow | — |
| **Language** | Python (desktop) / TypeScript (mobile) | — |

---

## 📁 Project Structure

```
PDP Project/
│
├── 📄 README.md                              # This file
├── 📄 SignBridge_Final_Report.md              # Detailed project report
│
├── 🐍 TRAINING & PREPROCESSING SCRIPTS
│   ├── 01_landmark_test.py                   # MediaPipe landmark testing
│   ├── 02_extract_landmarks.py               # Extract 63-feature vectors from 87K images
│   ├── 03_train_alphabet.py                  # Train Dense NN (A-Z classifier)
│   ├── SignBridge_Kaggle_Trainer.py           # Dynamic model trainer (Kaggle)
│   └── SignBridge_Kaggle_Trainer_Overhauled.py # Overhauled Conv1D+BiLSTM trainer
│
├── 🖥️ DESKTOP APPLICATION
│   ├── 09_final_signbridge.py                # ★ Final desktop app (run this)
│   ├── 07_desktop_app_final.py               # Earlier desktop iterations
│   ├── 04_realtime_inference.py              # Real-time testing script
│   └── 05_tts_speller.py                     # TTS + spelling integration
│
├── 🔄 MODEL CONVERSION
│   ├── 06_convert_final.py                   # H5 → TFLite conversion
│   ├── 06_convert_simple.py                  # Simplified conversion
│   └── 06_convert_tflite.py                  # TFLite with quantization
│
├── 📦 models/
│   ├── alphabet_model.h5                     # Trained alphabet model (289 KB)
│   ├── signbridge_dynamic_final.h5           # Trained dynamic model (4.7 MB)
│   ├── sign_model.tflite                     # Alphabet TFLite (24 KB)
│   ├── signbridge_dynamic.tflite             # Dynamic TFLite (503 KB)
│   ├── label_encoder.pkl                     # Alphabet label encoder
│   ├── label_map.json                        # Dynamic label mapping
│   ├── hand_landmarker.task                  # MediaPipe model
│   ├── confusion_matrix.png                  # Evaluation results
│   └── training_curves.png                   # Loss/accuracy plots
│
├── 📂 data/
│   ├── alphabet/
│   │   ├── landmarks.npy                     # 63,581 × 63 feature array (16 MB)
│   │   └── labels.npy                        # 63,581 string labels (1.8 MB)
│   ├── asl_alphabet_train/                   # Raw Kaggle images (87K)
│   └── dynamic/                              # Parquet sequence files
│
└── 📱 SignBridgeMobile/                      # React Native Mobile App
    ├── app/
    │   ├── _layout.tsx                       # Root layout
    │   └── index.tsx                         # ★ Main camera + UI screen (448 lines)
    ├── hooks/
    │   └── useHandTracking.ts                # ★ Core ML hook (562 lines)
    ├── components/
    │   └── SkeletonOverlay.tsx               # SVG hand skeleton renderer
    ├── utils/
    │   ├── ModelLoader.ts                    # TFJS model loading via bundleResourceIO
    │   ├── routerConstants.ts                # Hybrid router thresholds
    │   └── handConstants.ts                  # MediaPipe hand topology
    ├── assets/models/                        # Bundled TFJS model shards
    │   ├── alphabet/model.json + *.bin
    │   └── dynamic/model.json + *.bin
    ├── package.json
    └── app.json
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** (for desktop client and training)
- **Node.js 18+** and **npm** (for mobile client)
- **Kaggle Account** (for dataset download, if retraining)

### Desktop Client Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/SignBridge.git
cd SignBridge

# 2. Create and activate virtual environment
python3 -m venv signbridge_env
source signbridge_env/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install tensorflow opencv-python mediapipe numpy scikit-learn matplotlib seaborn

# 4. Run the desktop application
python 09_final_signbridge.py
```

**Desktop Keyboard Controls:**

| Key | Action |
|---|---|
| `C` | Clear all text |
| `P` | Speak the current sentence aloud (TTS) |
| `T` | Save transcript to `SignBridge_Transcript.txt` |
| `Q` / `Esc` | Quit application |

### Mobile Client Setup

```bash
# 1. Navigate to mobile directory
cd SignBridgeMobile

# 2. Install dependencies
npm install

# 3. Apply TensorFlow.js patches
npx patch-package

# 4. Start the Expo development server
npx expo start

# 5. Run on iOS (requires Xcode) or Android
npx expo run:ios
# or
npx expo run:android
```

> **Note:** The mobile app requires a physical device with a camera. Simulators/emulators will not work for real-time camera inference.

### Retraining Models (Optional)

```bash
# Download ASL Alphabet dataset
pip install kaggle
kaggle datasets download -d grassknoted/asl-alphabet
unzip asl-alphabet.zip -d data/

# Extract landmarks from images
python 02_extract_landmarks.py

# Train alphabet model
python 03_train_alphabet.py

# Train dynamic model (recommended: run on Kaggle with GPU)
# Upload SignBridge_Kaggle_Trainer_Overhauled.py to Kaggle Notebook

# Convert models
python 06_convert_final.py
```

---

## ⚙️ How It Works

### Step-by-Step Inference Pipeline

```
1. CAPTURE     Camera captures a video frame
                  ↓
2. DETECT      MediaPipe extracts 21 hand landmarks (63 features)
                  ↓
3. NORMALIZE   Wrist-relative normalization (wrist → origin)
                  ↓
4. ROUTE       Hybrid Router calculates wrist velocity
               ├── velocity ≤ 0.025 → STATIC path (Alphabet model)
               └── velocity > 0.025 → DYNAMIC path (buffer → LSTM)
                  ↓
5. PREDICT     Selected model outputs class probabilities
                  ↓
6. FILTER      Confidence thresholding + consistency buffer
                  ↓
7. OUTPUT      Letter/phrase added to sentence → optional TTS
```

### Memory Management (Mobile)

The mobile client implements strict tensor lifecycle management to prevent memory leaks:

```typescript
// Every frame: dispose input tensor in finally block
try {
  const hands = await detector.estimateHands(imageTensor);
  // ... process hands ...
} finally {
  tf.dispose(imageTensor);  // Always dispose, even on error
}

// Model predictions: auto-dispose via tf.tidy()
const result = tf.tidy(() => {
  const input = tf.tensor2d([flatFrame], [1, 63]);
  const output = model.predict(input);
  return output.dataSync();  // Read data before tidy disposes tensors
});
```

---

## 💡 Novel Contributions

### 1. Momentum-Based Hybrid Routing
Algorithmically detects user intent (spelling vs. phrase signing) via Euclidean wrist velocity tracking with a motion-patience buffer — **no manual mode toggle required**.

### 2. Wrist-Relative Spatial Normalization
All 21 landmarks normalized relative to wrist position, achieving translation and scale invariance — solves the "Floating Hand" problem.

### 3. Optimized Mobile Edge Inference
Successfully bridged native camera → TF.js tensors with `cameraWithTensors` HOC, frame skipping, 192×192 resize, and strict `tf.dispose()` — achieving 15–24 FPS with zero memory leaks.

### 4. Dual-Model Sharded Deployment
Both models bundled as TFJS sharded weights (`model.json` + `.bin`) inside the React Native Metro bundle — fully offline, no download required.

---

## 🔮 Future Roadmap

- [ ] **NLP Auto-Correction** — Spell-check via edit distance (TextBlob/SymSpell)
- [ ] **Predictive Text** — Context-aware next-word suggestions from local LLM
- [ ] **Self-Calibrating Router** — Adaptive velocity threshold based on user's baseline jitter
- [ ] **Expanded Vocabulary** — Scale from 21 to 50–100 dynamic phrases
- [ ] **Two-Handed Recognition** — Extend to 126 features (2 hands × 63 features)
- [ ] **Android Optimization** — Platform-specific performance tuning

---

## 👥 Team

| Member | Role | Key Contributions |
|---|---|---|
| **Mahesh** | Lead AI & System Architect | Conv1D+BiLSTM architecture, Hybrid Router algorithm, wrist normalization, `useHandTracking.ts` tensor bridge |
| **Vishwesh** | Data Engineering | Dataset extraction (87K images → 63K samples), padding/truncation algorithms, confusion matrices |
| **Manoj** | Frontend & UI Design | Glassmorphism mobile UI, SVG skeleton overlay (`SkeletonOverlay.tsx`), OpenCV desktop overlays |
| **Dhananjayalu** | Deployment & QA | H5→TFJS model conversion, latency calibration, Expo SDK 54 patches, TTS integration |

---

## 📚 References

| Resource | Link |
|---|---|
| ASL Alphabet Dataset | https://www.kaggle.com/datasets/grassknoted/asl-alphabet |
| ASL Signs Competition | https://www.kaggle.com/competitions/asl-signs/data |
| MediaPipe Hands | https://developers.google.com/mediapipe/solutions/vision/hand_landmarker |
| TensorFlow.js (React Native) | https://www.tensorflow.org/js/guide/react_native |
| Expo SDK 54 | https://docs.expo.dev/ |
| TensorFlow Keras | https://www.tensorflow.org/api_docs/python/tf/keras |

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>SignBridge</b> — Breaking communication barriers with on-device AI 🤟
</p>
