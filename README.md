🩺 Diabetic Retinopathy Detection Using Machine Learning

An end-to-end AI-powered medical imaging system for detecting and grading Diabetic Retinopathy (DR) using OCT and Fundus retinal images. The project combines deep learning–based feature extraction with classical machine learning to deliver accurate, interpretable, and efficient diagnosis support.

📌 Project Overview

This project presents a hybrid machine learning framework that leverages a pretrained VGG16 CNN for deep feature extraction and LightGBM for classification. The system analyzes retinal images to classify patients into Normal, Moderate DR, and Severe DR stages. The approach focuses on high accuracy, reduced computational cost, and model interpretability, making it suitable for real-world clinical and tele-ophthalmology applications.

🚀 Features

🧠 Hybrid ML Architecture (Deep Learning + Classical ML)
🩻 Supports OCT and Fundus retinal images
📊 High accuracy (up to 99% with LightGBM)
🔍 Interpretable predictions using feature-based learning
⚡ Fast inference with low computational overhead
📈 Detailed evaluation (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
☁️ Cloud-ready (Google Colab / Jupyter compatible)

🛠️ Technologies Used

->Programming Language:

  Python 3.x

->Deep Learning & ML:

  TensorFlow / Keras (VGG16 feature extractor)
  
  LightGBM (classification)
  
  Scikit-learn (evaluation & preprocessing)

->Image Processing:

  OpenCV
  
  NumPy

->Visualization:

  Matplotlib
  Seaborn
  
->Environment:

Google Colab / Jupyter Notebook

📂 Project Structure

diabetic-retinopathy-detection/

│

├── dataset/

│   ├── OCT/

│   └── Fundus/

│

├── notebooks/

│   └── Diabetic_Retinopathy.ipynb

│

├── models/

│   ├── vgg16_feature_extractor.pkl

│   └── lightgbm_classifier.pkl

│

├── results/

│   ├── confusion_matrix.png

│   └── accuracy_plots.png

│

├── requirements.txt

└── README.md


⚙️ How It Works

1️⃣ User provides OCT or Fundus retinal images

2️⃣ Images are resized, normalized, and preprocessed

3️⃣ VGG16 CNN extracts deep visual features

4️⃣ Features are standardized and passed to LightGBM

5️⃣ Model predicts the Diabetic Retinopathy stage

6️⃣ Output includes prediction label and confidence score

🧪 How to Run Locally

Clone the repository

git clone https://github.com/mloukikreddy/diabetic-retinopathy.git


Navigate to the project directory

cd diabetic-retinopathy-detection


Install dependencies

pip install -r requirements.txt


Run the notebook

jupyter notebook


Open Diabetic_Retinopathy.ipynb and execute all cells.


🎯 Learning Outcomes:-

✔ Medical image preprocessing using OpenCV

✔ Deep feature extraction with pretrained CNNs

✔ Hybrid ML model design (DL + LightGBM)

✔ Model evaluation using clinical performance metrics

✔ Building interpretable and scalable AI healthcare systems


👤 Authors:-

Loukik Reddy Mekala
📌 GitHub: https://github.com/mloukikreddy

Project Domain:
Artificial Intelligence | Machine Learning | Medical Image Analysis
