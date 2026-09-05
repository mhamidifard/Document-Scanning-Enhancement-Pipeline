<div align="center">
  <h1>📸 Document Scanning & Enhancement Pipeline</h1>
  <p>An End-to-End Deep Learning pipeline for Document Corner Detection, Rectification, and Visual Enhancement.</p>
</div>

---

## 📖 Overview

This project provides a robust, PyTorch-based pipeline designed to automatically detect document corners, rectify the perspective, and enhance the visual quality of scanned documents (e.g., removing shadows, noise, and lighting variations). The pipeline synthesizes its own training data on-the-fly using `kornia`, making it highly robust to real-world degradations without requiring massive manually annotated datasets.

Similar to popular applications like CamScanner, this project bridges the gap between raw camera photos and clean, flat, high-contrast document scans suitable for Optical Character Recognition (OCR) and archiving.

## ✨ Key Features

- **Synthetic Data Generation Engine**: Utilizes `kornia` for on-the-fly, differentiable augmentations. Generates realistic degradations including dynamic shadows, noise, color jitter, and distractors (thumbs, spiral bindings).
- **Document Corner Detection**: Two state-of-the-art approaches:
  - **Regression-based**: Predicts normalized `(x, y)` corner coordinates directly.
  - **Heatmap-based**: Predicts Gaussian heatmaps for each corner for higher precision.
- **Document Enhancement**: A customized UNet architecture that removes shadows and noise, restoring the document to a clean, flat state.
- **End-to-End Evaluation**: Includes built-in evaluation scripts that measure traditional metrics (PSNR, SSIM) as well as End-to-End OCR performance using `Tesseract`.
- **Inference Ready**: Easy-to-use scripts for both corner detection and image enhancement on real-world photos.

## 🏗️ Project Structure

```text
.
├── src/
│   ├── dataset.py               # Data loading for clean documents and backgrounds
│   ├── model.py                 # Neural Network architectures (UNet, RegressionNet, HeatmapNet)
│   ├── pipeline.py              # On-the-fly synthetic degradation and warping using Kornia
│   ├── train.py                 # Training loop for the Enhancement model
│   ├── train_corner.py          # Training loop for Corner Detection models
│   ├── evaluate*.py             # Evaluation scripts (PSNR/SSIM, OCR, Real Data)
│   ├── inference*.py            # Inference scripts for real-world images
│   └── visualize_pipeline.py    # Tools for visualizing the synthetic data generation
├── data/                        # Directory for datasets (clean documents, backgrounds, real photos)
├── result/                      # Directory for saving outputs, plots, and checkpoints
├── cv.ipynb                     # Jupyter Notebook demonstrating training and evaluation in Colab
├── pyproject.toml               # Project metadata and dependencies
└── uv.lock                      # Lockfile for reproducible builds
```

## 🚀 Setup & Installation

This project uses modern Python packaging. Make sure you have Python 3.14+ (or compatible version as specified in your setup) and an environment manager like `uv` or `pip`.

```bash
# Clone the repository
git clone <your-repo-url>
cd Document-Scanning---Enhancement

# Install dependencies using pip
pip install -e .

# Or using uv (recommended for fast resolution)
uv sync
```

### System Requirements
To run the End-to-End OCR evaluation, you need to install Tesseract on your system:
- **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`
- **macOS:** `brew install tesseract`
- **Windows:** Download the installer from the [UB-Mannheim Tesseract project](https://github.com/UB-Mannheim/tesseract/wiki).

## 🧠 Training

The project generates synthetic training data dynamically by overlaying clean documents onto random backgrounds and applying severe perspective warping and photometric degradations.

### 1. Train Corner Detection
You can choose between the `regression` and `heatmap` approaches:
```bash
# Heatmap approach (Recommended)
python src/train_corner.py --approach heatmap --epochs 50 --img_size 256 --use_dropout

# Regression approach
python src/train_corner.py --approach regression --epochs 50 --img_size 256
```

### 2. Train Document Enhancement
Trains the UNet to map heavily degraded and warped document patches back to their clean, flat ground truths.
```bash
python src/train.py --epochs 100 --batch_size 8 --img_size 768 --lr 0.0001
```

## 📊 Evaluation

Evaluate the models on your validation/test sets to compute quantitative metrics.

**Enhancement Evaluation (PSNR / SSIM):**
```bash
python src/evaluate.py --img_size 768
```

**Corner Detection Evaluation (Real Dataset):**
```bash
python src/evaluate_real.py --dataset_dir data/real_test_photos
```

**End-to-End OCR Evaluation:**
Measures how well the entire pipeline improves text recognition accuracy compared to raw images.
```bash
python src/evaluate_e2e.py \
    --dataset_dir data/real_test_photos \
    --corner_model_dir checkpoints_corner \
    --enh_model_dir checkpoints
```

## 🔍 Inference

Apply the trained models to your own photos!

**Enhance a cropped document:**
```bash
python src/inference_enhancement.py \
    --input data/real_croped/2.jpg \
    --output result/enhancement_output/2_enhanced.jpg
```

**Detect corners on a raw photo:**
```bash
python src/inference_corner.py \
    --approach heatmap \
    --input data/real_test_photos/1.jpg \
    --output result/regression/1_corners.jpg
```

## 👨‍💻 Author

- **Mohammad Sajjad** - [mohammadsajjad.hamidifard@email.kntu.ac.ir](mailto:mohammadsajjad.hamidifard@email.kntu.ac.ir)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
