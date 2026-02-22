# 🛰️ Satellite Change Detector

A deep learning-based system designed to detect structural and environmental changes between satellite images captured at different time intervals. This project was developed as a technical assessment for an Artificial Intelligence Engineer position in the defense industry.

## 📌 Project Overview
The system identifies changes such as new urban developments, natural disaster impacts, and agricultural shifts by comparing "before" and "after" satellite imagery. It utilizes a U-Net architecture optimized for semantic segmentation tasks.

## 🚀 Key Features
* Robust Data Pipeline: Implemented using Albumentations for seamless data augmentation (flips, rotations) and consistent resizing across dual-image inputs.
* 6-Channel Input Processing: The model processes 3-channel "before" and 3-channel "after" images simultaneously to capture spatial differences effectively.
* Advanced Loss Function: Utilizes a combination of Binary Crossentropy and Dice Loss to handle class imbalance often found in satellite change detection.
* Model Persistence: Features automated ModelCheckpointing to save the best weights (best_model.h5) and EarlyStopping to prevent overfitting.
* Inference Module: Supports loading pre-trained weights for instant change detection without retraining.

## 🧠 Technical Stack
* Language: Python 3.11
* Deep Learning: TensorFlow / Keras
* Image Processing: OpenCV, Albumentations
* Visualization: Matplotlib, NumPy

## 📁 Dataset Structure
The model is trained on the CDD (Change Detection Dataset).
dataset/
├── train/
│   ├── A/         → After images
│   ├── B/         → Before images
│   └── out/       → Change masks (Ground Truth)
├── val/
└── test/

## 🛠️ Installation & Usage
1. Clone the repository:
   git clone https://github.com/ysuffkibarr/SatelliteChangeDetector.git
   cd SatelliteChangeDetector

2. Install dependencies:
   pip install tensorflow opencv-python matplotlib numpy albumentations

3. Run the application:
   python app.py
   Note: If a pre-trained best_model.h5 exists, the script will skip training and perform inference immediately.

## 📊 Results
The model generates a binary change mask (predicted_change.png) highlighting the detected differences between the input image pairs.