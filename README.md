# ✏️ Draw & Guess — Sketch Classifier

[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![QuickDraw](https://img.shields.io/badge/Dataset-QuickDraw-4285F4?logo=google&logoColor=white)](https://quickdraw.withgoogle.com/data)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-FFD21E?logo=gradio&logoColor=black)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

A ResNet-style CNN trained on Google's QuickDraw dataset that recognises hand-drawn doodles across **30 categories**. Draw anything in the Gradio sketchpad and the model guesses what it is in real time — with a confidence threshold so it says "not sure" instead of guessing wrong.

---

## 📊 Performance Benchmark

| Metric | Score | Strategy |
| :--- | :--- | :--- |
| **Test Accuracy** | **~87–91%** | ResNet skip connections + 10k samples/class |
| **Final Test Loss** | **< 0.35** | Sparse categorical cross-entropy |
| **Training Duration** | ~7–10 minutes | GPU Accelerated (T4 Runtime) |
| **Chance Level** | 3.3% | Random guessing across 30 classes |

---

## 🧠 Model Architecture

Uses the Keras Functional API with residual (skip connection) blocks.

```text
      [ Input: 28×28×1 ]
             │
      Conv2D (32) + BN + ReLU
             │
    ┌────────────────────┐
    │  Residual Block    │  32 filters
    │  Conv→BN + Conv→BN │
    │  └── Add(x, skip)  │
    └────────┬───────────┘
             │
    ┌────────────────────┐
    │  Residual Block    │  64 filters, stride 2 (downsample)
    │  Conv→BN + Conv→BN │
    │  └── Add(x, skip)  │
    └────────┬───────────┘
             │
    ┌────────────────────┐
    │  Residual Block    │  64 filters
    └────────┬───────────┘
             │
    ┌────────────────────┐
    │  Residual Block    │  128 filters, stride 2 (downsample)
    └────────┬───────────┘
             │
    ┌────────────────────┐
    │  Residual Block    │  128 filters
    └────────┬───────────┘
             │
    GlobalAveragePooling2D
    Dense (256, ReLU) + Dropout (0.5)
    Dense (30, Softmax)
```

**Why skip connections?** Adding the block's input directly to its output — `output = F(x) + x` — lets gradients flow backward through the shortcut path. This solves the vanishing gradient problem that makes deep networks hard to train, and consistently improves accuracy over plain CNNs.

---

## 🗂️ Dataset

**QuickDraw** by Google — 50M+ hand-drawn doodles across 345 categories.

This project uses 30 categories × 10,000 drawings = **300,000 total samples**.

| Split | Samples |
|-------|---------|
| Train (70%) | 210,000 |
| Validation (15%) | 45,000 |
| Test (15%) | 45,000 |

**30 categories:**
cat · dog · car · house · tree · sun · fish · bird · bicycle · airplane · pizza · umbrella · clock · star · apple · flower · mountain · boat · guitar · hat · lion · elephant · butterfly · mushroom · rainbow · truck · cake · moon · eye · spider

---

## ✨ Key Features

- **ResNet-style skip connections** — residual blocks with `Add()([x, shortcut])` let gradients flow freely through deep layers, producing better accuracy than a plain Sequential CNN on the same data.
- **Keras Functional API** — skip connections require branching paths which can't be expressed in `Sequential`. The Functional API (`Model(inputs, outputs)`) enables any graph-shaped architecture.
- **Custom dataset assembly** — downloads 30 `.npy` files from Google Cloud Storage, samples 10,000 drawings from each, and stacks them into a single dataset.
- **Sparse categorical cross-entropy** — labels stay as integers (0–29) instead of one-hot vectors. More memory-efficient for multi-class problems with many categories.
- **Mean image visualization** — shows the average of all 10,000 drawings per category. Blurry = high variation in how people draw it (harder). Sharp = drawn consistently (easier).
- **Per-class accuracy bar chart** — bars colored green (above average) or red (below average), with a dashed mean line. Much more readable than a 30×30 confusion matrix.
- **Confidence threshold** — if the model's top prediction is below 40%, the Gradio app shows "Not sure — keep drawing!" instead of a wrong confident answer.
- **Image inversion** — QuickDraw stores white strokes on black. The sketchpad draws black on white. The preprocessing pipeline inverts before inference.
- **ReduceLROnPlateau + Early stopping** — automatic learning rate scheduling and best-weight restoration.

---

## 🛠️ Tech Stack

- **Neural network:** TensorFlow 2.x & Keras (Functional API)
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
4. Step 2 downloads 30 files (~5–6 min), then training begins
5. Scroll to the last cell and draw in the Gradio sketchpad

### Local Machine

```bash
pip install tensorflow gradio seaborn pillow scikit-learn

jupyter notebook quickdraw_sketch_classifier.ipynb
```

---

## 💡 What I Learned

- **ResNet skip connections** — why `output = F(x) + x` solves vanishing gradients
- **Keras Functional API** — building graph-shaped models that Sequential can't express
- **Multi-class classification** — 30-way softmax vs binary sigmoid
- **Sparse categorical cross-entropy** — integer labels vs one-hot encoding
- **Custom dataset assembly** — downloading, sampling, and stacking `.npy` files
- **Confidence thresholds** — how to make a model say "I don't know" instead of guessing wrong
- **Per-class accuracy analysis** — identifying which categories a model struggles with and why

---

## 👤 Author

**Mohamed Ouledali** — Engineering Student
