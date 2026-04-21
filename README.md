# Diabetic Retinopathy Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mloukikreddy/Diabetic-Retinopathy/blob/main/Diabetic_Retinopathy.ipynb)

---

## Overview

This project implements a deep learning-based approach to detect diabetic retinopathy from retinal fundus images. Early detection is critical in preventing vision loss, and this model aims to assist in automated screening using image classification techniques.

---

## Problem Statement

Diabetic retinopathy is a leading cause of blindness among diabetic patients. Manual diagnosis is time-consuming and requires expert ophthalmologists. This project explores deep learning techniques to automate detection using retinal images.

---

## Dataset

- **Source:** https://www.kaggle.com/datasets/danielwill004/new-dr-dataset  
- **Type:** Retinal fundus images  
- **Size:** ~5000 images  
- **Classes:** Multiple stages of diabetic retinopathy  

> ⚠️ Dataset is not included due to large size.

---

## Project Workflow

1. Data Collection  
2. Data Preprocessing (resizing, normalization)  
3. Train-validation split  
4. CNN Model Development  
5. Model Training  
6. Evaluation & Prediction  

---

## Model Architecture

- Convolutional Neural Network (CNN)  
- Conv2D + MaxPooling layers  
- Fully connected layers  
- Dropout for regularization  

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- OpenCV  
- Google Colab  

---

## Setup & Execution

### Option 1 — Google Drive

1. Download dataset  
2. Upload to Google Drive  
3. Mount Drive in Colab  
4. Update dataset path  

---

### Option 2 — Kaggle API (Recommended)

```python
!pip install -q kaggle
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d danielwill004/new-dr-dataset
!unzip new-dr-dataset.zip -d dataset/

```
---

## Results
Performance Metrics
Accuracy: 100.00%
Precision: 1.00
Recall: 1.00
F1-Score: 1.00

Note: High accuracy may indicate potential overfitting; further validation is recommended.
---

## Classification Report

| Class      | Precision | Recall | F1-Score |
| ---------- | --------- | ------ | -------- |
| 0 (Normal) | 1.00      | 1.00   | 1.00     |
| 1 (DR)     | 1.00      | 1.00   | 1.00     |

---

## Sample Prediction
Prediction: Moderate Diabetic Retinopathy
Confidence: 99.50%

---

## Key Highlights
Used transfer learning (VGG16) for feature extraction
Combined with LightGBM for classification
Achieved high accuracy on medical image dataset
Supports automated retinal image analysis

---

## Limitations
Performance depends on dataset quality
Requires further tuning for higher accuracy
Not suitable for clinical use without validation

---

## Future Improvements
Use larger and more diverse datasets
Apply cross-validation
Implement explainable AI (Grad-CAM)
Deploy as a web application
Use advanced models (EfficientNet, ResNet)

---

## Reproducibility
Dataset source provided
Kaggle API supported
Lightweight repository

---

## Notes
Developed using Google Colab
Dataset handled via Google Drive
Large files excluded using .gitignore
