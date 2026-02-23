# 🛰️ AI Satellite Change Detector

A deep learning-based system designed to detect structural and environmental changes between satellite images captured at different time intervals. This project was developed as a technical assessment for an Artificial Intelligence Engineer position in the defense industry and features a production-ready interactive web interface.

## 📌 Project Overview
The system identifies changes such as new urban developments, natural disaster impacts, and agricultural shifts by comparing "before" and "after" satellite imagery. It utilizes an advanced Siamese ResNet50 U-Net architecture, leveraging transfer learning for highly accurate feature extraction.

## 🚀 Key Features
* Interactive Web Interface: A sleek, user-friendly frontend built with Gradio for real-time change detection inference.
* Siamese ResNet50 Backbone: Processes "before" and "after" images through identical ResNet50 networks (pre-trained on ImageNet) before feeding features into the U-Net decoder.
* Robust Data Pipeline: Implemented using Albumentations for seamless data augmentation (flips, rotations) and consistent resizing.
* Advanced Loss Function: Utilizes a combination of Binary Crossentropy and Dice Loss to handle class imbalance.
* Model Persistence: Features automated ModelCheckpointing and EarlyStopping.
* Post-Processing: Applies OpenCV morphological operations (Opening/Closing) to the output mask to remove noise and deliver a clean, crisp prediction.

## 🧠 Technical Stack
* Language: Python 3.11
* Deep Learning: TensorFlow / Keras (ResNet50)
* Web UI: Gradio
* Image Processing: OpenCV, Albumentations
* Visualization: Matplotlib, NumPy

## 📁 Dataset Structure
The model is trained on the CDD (Change Detection Dataset). Note: The dataset is excluded from version control to maintain repository performance.

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
   pip install tensorflow opencv-python matplotlib numpy albumentations gradio

3. Run the Web Interface (Inference Mode):
   python ui.py
   *This will launch a local web server (typically at http://127.0.0.1:7860) where you can upload image pairs and see the predicted change mask instantly.*

4. Train the Model (Optional):
   python app.py
   *If a pre-trained best_model_resnet50.h5 does not exist, this will initiate the training pipeline from scratch.*

## 📬 Contact
Yusuf Kibar
* Computer Programming Student at Selçuk University Ilgın Vocational School
* Passionate about AI, Software Development, and Cybersecurity
* Led a TÜBİTAK project
* Developer of BullEye (stock tracking), SymptoCheckAI (disease prediction), and AeroWave (weather forecasting)
* Goal: Establishing a software company and achieving financial independence