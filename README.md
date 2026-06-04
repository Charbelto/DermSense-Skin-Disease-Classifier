# DermSense AI: Dermoscopic Skin Lesion Classification System

DermSense AI is an end-to-end medical computer vision system that classifies dermatological skin lesions into seven distinct diagnostic categories using the **HAM10000 dataset**. The project features a comparative study between a traditional Machine Learning baseline (SVM with handcrafted HOG & Color features) and a Deep Learning model (EfficientNet-B0 transfer learning in PyTorch), supplemented by post-hoc model explainability (Grad-CAM) and a polished, responsive web application interface.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset & Classes](#-dataset--classes)
3. [System Architecture](#-system-architecture)
4. [Machine Learning Models](#-machine-learning-models)
5. [Model Explainability (Grad-CAM)](#-model-explainability-grad-cam)
6. [Interactive Web Dashboard](#-interactive-web-dashboard)
7. [Installation & Usage](#-installation--usage)

---

## 🎯 Project Overview
Melanoma and other skin cancers are highly treatable if diagnosed early. However, manual dermoscopic screening is time-consuming and subjective. This project implements a complete computer vision pipeline designed to:
1. **Benchmark Traditional vs. Deep Learning**: Evaluate whether handcrafted visual descriptors (HOG + Color histograms) can compete with hierarchical convolutional features (EfficientNet-B0).
2. **Handle Extreme Class Imbalance**: Utilize Weighted Random Sampling to address minority class representation (such as dermatofibromas).
3. **Clinical Explainability**: Expose model attention regions using Grad-CAM so practitioners can audit what features drove the prediction.
4. **Interactive Dashboard**: Package the system into a modern web client interface representing the clinical front-end.

---

## 📊 Dataset & Classes

The system is trained on the **HAM10000 Dataset** (Human Against Machine), which contains 10,015 dermatoscopic images confirming seven diagnostic categories:

*   **akiec**: Actinic Keratoses (Pre-cancerous)
*   **bcc**: Basal Cell Carcinoma (Malignant)
*   **bkl**: Benign Keratosis (Benign)
*   **df**: Dermatofibroma (Benign)
*   **mel**: Melanoma (Malignant)
*   **nv**: Melanocytic Nevi (Benign)
*   **vasc**: Vascular Lesions (Benign)

---

## 🛠 System Architecture

The codebase is organized into a clean, modular structure:

```text
DermSense-Skin-Disease-Classifier/
├── docs/
│   └── academic/            # Coursework guidelines, briefs, and reflection report
├── samples/                 # Sample images for testing inference
├── outputs/                 # Saved training graphs, confusion matrices, and Grad-CAMs
├── src/                     # Core Package Modules
│   ├── __init__.py
│   ├── config.py            # Global hyperparameters, paths, and device parameters
│   ├── dataset.py           # Remote download, class folders sorter, dataloaders
│   ├── svm_model.py         # Baseline HOG + Color feature extraction and SVM training
│   ├── deep_learning.py     # EfficientNet-B0 network layers, loss, and training loop
│   ├── explainability.py    # Grad-CAM hooks and saliency map generator
│   └── utils.py             # Visual charts generators (ROC curves, confusion heatmaps)
├── train_svm.py             # Train baseline SVM classifier
├── train_cnn.py             # Train & fine-tune PyTorch CNN classifier
├── explain.py               # Generate Grad-CAM overlays on test images
├── Skin_Disease_Classification.ipynb # Jupyter notebook demonstration walkthrough
├── index.html               # Web interface HTML5 structure
├── style.css                # Web interface CSS styling (Midnight Medical theme)
├── script.js                # Web interface client logic (TensorFlow.js metrics)
├── requirements.txt         # Package dependencies
└── README.md                # System documentation
```

---

## 🧠 Machine Learning Models

### 1. Traditional ML Baseline (SVM + HOG)
*   **Feature Extraction**: Images are resized to $128 \times 128$ pixels. We extract **Color Histograms** (32 bins per RGB channel) and **Histogram of Oriented Gradients (HOG)** features (orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2)).
*   **Model**: Support Vector Classifier (SVC) with a Radial Basis Function (RBF) kernel, optimized with cost parameter $C=10$ and balanced class weights.

### 2. Deep Learning Classifier (EfficientNet-B0)
*   **Data Augmentations**: Horizontal/vertical flips, random rotations, color jitter (brightness, contrast, saturation, hue), and affine transformations are applied to prevent overfitting.
*   **Architecture**: EfficientNet-B0 pretrained on ImageNet, with custom dropout (0.3) and dense layers replaced at the head.
*   **Training Strategy**:
    *   **Phase 1**: Frozen backbone, training the classifier head for 20 epochs (learning rate $10^{-4}$).
    *   **Phase 2**: Unfreezing the entire network for fine-tuning (learning rate $5 \times 10^{-5}$) with Plateau scheduler decay and Early Stopping.

---

## 🔍 Model Explainability (Grad-CAM)

To provide transparency, post-hoc explanations are generated using **Grad-CAM** (Gradient-weighted Class Activation Mapping). The tool intercepts the forward activations and backward gradients of the final convolutional layer of EfficientNet (`model.features[-1]`). 

It outputs a normalized heat map highlighting the precise lesions textures and structural boundary cues that influenced the final classification decision, showing:
1. **Original Image**: Raw dermatoscopic scan.
2. **Grad-CAM Overlay**: Thermal attention map highlighting high-contrast clinical boundaries.

---

## 🎨 Interactive Web Dashboard

The project includes **DermSense**, a midnight-medical-themed web client allowing real-time simulated dermatoscopic triage:

*   **Analysis Section**: Drag-and-drop zone with scanner animations, processing indicators, and interactive probability charts for the 7 disease categories.
*   **Scan History**: Review past logs, classifications, and diagnostic metrics.
*   **Clinical Reports**: Pre-formatted cards to download clinical reports.
*   **System Settings**: Configure AI sensitivity thresholds and toggle themes.

*Note: The frontend code uses TensorFlow.js to process images in the browser and runs in a demo mode using tensor statistics (image mean/variance).*

---

## 🚀 Installation & Usage

### 1. Environment Setup
```bash
# Clone the repository
git clone <repo-url>
cd DermSense-Skin-Disease-Classifier

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install PyTorch and package dependencies
pip install -r requirements.txt
```

### 2. Train and Evaluate SVM Baseline
To run feature extraction, train the SVM classifier, and save baseline metrics:
```bash
python train_svm.py
```

### 3. Train and Fine-Tune CNN Model
To download HAM10000 dataset zip files programmatically, split folders, and run the two-phase training process:
```bash
python train_cnn.py
```
*Note: Toggle `RUN_PROFILE = 'full'` in `src/config.py` to run full 40 epochs; by default it runs `repro` (4 epochs total) for quick execution verification.*

### 4. Run Explainability Analysis
To run Grad-CAM maps on random test set items:
```bash
python explain.py
```
Outputs and graphs will be written to `outputs/`.
