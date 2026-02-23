import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf
import os

def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

print("Yapay Zeka Motoru Yükleniyor...")
model_path = "best_model_resnet50.h5"
if os.path.exists(model_path):
    model = load_model(model_path, custom_objects={'combined_loss': combined_loss, 'dice_loss': dice_loss})
    print("Motor Hazır!")
else:
    model = None
    print("UYARI: Model bulunamadı. Lütfen önce app.py dosyasını çalıştırarak modeli eğitin.")

def predict_change(img_before, img_after):
    if model is None:
        return np.zeros((256, 256))

    img1_resized = cv2.resize(img_before, (256, 256)) / 255.0
    img2_resized = cv2.resize(img_after, (256, 256)) / 255.0

    input_img = np.concatenate([img1_resized, img2_resized], axis=-1)
    input_img = np.expand_dims(input_img, 0)

    # Tahmin
    pred_mask = model.predict(input_img)[0, :, :, 0]
    binary_mask = (pred_mask > 0.3).astype(np.uint8) * 255

    kernel = np.ones((5,5), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    return cleaned_mask

interface = gr.Interface(
    fn=predict_change,
    inputs=[
        gr.Image(label="Before Image (Önceki Uydu Görüntüsü)"),
        gr.Image(label="After Image (Sonraki Uydu Görüntüsü)")
    ],
    outputs=gr.Image(label="Detected Change (Tespit Edilen Değişim)"),
    title="🛰️ AI Satellite Change Detector",
    description="Upload two satellite images from different times to detect structural and environmental changes. Powered by Siamese ResNet50 U-Net architecture.",
    theme="default",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()