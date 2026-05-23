# ✏️ Draw & Guess — Sketch Classifier

[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![QuickDraw](https://img.shields.io/badge/Dataset-QuickDraw-4285F4?logo=google&logoColor=white)](https://quickdraw.withgoogle.com/data)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-FFD21E?logo=gradio&logoColor=black)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

A CNN trained on Google's QuickDraw dataset that recognises hand-drawn doodles across 15 categories. Draw anything in the Gradio sketchpad and the model guesses what it is in real time.

---

## 📊 Performance Benchmark

| Metric | Score | Strategy |
| :--- | :--- | :--- |
| **Test Accuracy** | **~85–88%** | Triple-block CNN + augmentation |
| **Final Test Loss** | **< 0.40** | Sparse categorical cross-entropy |
| **Training Duration** | ~5–7 minutes | GPU Accelerated (T4 Runtime) |
| **Chance Level** | 6.7% | Random guessing across 15 classes |

---

## 🧠 Model Architecture

```text
        [ Input: 28×28×1 grayscale sketch ]
                       │
      [ tf.data augmentation pipeline ]
      ├── Random Rotation  (±10%)
      ├── Random Zoom      (±10%)
      └── Random Translation (±10%)
                       │
   ┌──────────────────────────────────────┐
   │ Conv Block 1 — stroke detection      │
   │ ├── Conv2D (32, 3×3, ReLU, Same)     │
   │ ├── BatchNormalization               │
   │ ├── Conv2D (32, 3×3, ReLU, Same)     │
   │ ├── BatchNormalization               │
   │ └── MaxPooling2D + Dropout (0.25)    │
   └──────────────────┬───────────────────┘
                      │
   ┌──────────────────────────────────────┐
   │ Conv Block 2 — shape detection       │
   │ ├── Conv2D (64, 3×3, ReLU, Same)     │
   │ ├── BatchNormalization               │
   │ ├── Conv2D (64, 3×3, ReLU, Same)     │
   │ ├── BatchNormalization               │
   │ └── MaxPooling2D + Dropout (0.25)    │
   └──────────────────┬───────────────────┘
                      │
   ┌──────────────────────────────────────┐
   │ Conv Block 3 — object structure      │
   │ ├── Conv2D (128, 3×3, ReLU, Same)    │
   │ ├── BatchNormalization               │
   │ └── Dropout (0.25)                   │
   └──────────────────┬───────────────────┘
                      │
   ┌──────────────────────────────────────┐
   │ Classification Head                  │
   │ ├── Flatten                          │
   │ ├── Dense (256, ReLU)                │
   │ ├── BatchNormalization               │
   │ ├── Dropout (0.5)                    │
   │ └── Dense (15, Softmax)              │
   └──────────────────────────────────────┘
```

---

## 🗂️ Dataset

**QuickDraw** by Google — 50M+ hand-drawn doodles across 345 categories.

This project uses 15 categories × 5,000 drawings = **75,000 total samples**.

| Split | Samples |
|-------|---------|
| Train (70%) | 52,500 |
| Validation (15%) | 11,250 |
| Test (15%) | 11,250 |

**15 categories:** cat · dog · car · house · tree · sun · fish · bird · bicycle · airplane · pizza · umbrella · clock · star · apple

---

## ✨ Key Features

- **Custom dataset assembly** — downloads 15 `.npy` files from Google Cloud Storage and samples from each, rather than loading a pre-packaged dataset.
- **Sparse categorical cross-entropy** — labels stay as integers (0–14) instead of one-hot vectors. More memory-efficient for multi-class problems with many categories.
- **Image inversion in Gradio** — QuickDraw stores white strokes on black. The sketchpad draws black on white. The preprocessing pipeline inverts before inference so the model sees what it was trained on.
- **Live predictions** — Gradio app updates top 3 guesses as you draw, before you even lift your finger.
- **ReduceLROnPlateau** — halves the learning rate after 2 epochs without val loss improvement.
- **Early stopping** — restores best weights automatically after 5 epochs of no improvement.
- **Full diagnostics** — 15×15 confusion matrix, classification report, and a grid of the sketches the model got wrong.

---

## 🛠️ Tech Stack

- **Neural network:** TensorFlow 2.x & Keras
- **Data pipeline:** tf.data with parallel augmentation + prefetch
- **UI:** Gradio
- **Image processing:** Pillow (PIL), NumPy
- **Evaluation:** scikit-learn, Matplotlib, Seaborn

---

## 🚀 How to Run

### Google Colab (Recommended)

1. Upload `quickdraw_sketch_classifier.ipynb` to [Google Colab](https://colab.research.google.com)
2. Enable GPU: **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Run all** (`Ctrl+F9`)
4. Step 2 downloads the dataset (~3 min), then training begins
5. Scroll to the last cell and draw in the sketchpad

### Local Machine

```bash
pip install tensorflow gradio seaborn pillow scikit-learn

jupyter notebook quickdraw_sketch_classifier.ipynb
```

---

## 💡 What I Learned

- **Multi-class classification** — 15-way softmax vs binary sigmoid
- **Sparse categorical cross-entropy** — integer labels vs one-hot encoding
- **Custom dataset assembly** — downloading, sampling, and stacking `.npy` files
- **tf.data pipeline** — parallel augmentation and prefetching for fast GPU feeding
- **Confusion matrix at scale** — reading a 15×15 matrix to find which categories the model confuses most (cat vs dog is a classic)

---

## 👤 Author

**Mohamed Ouledali** — Engineering Student
