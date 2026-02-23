import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow.keras.backend as K
import tensorflow as tf
import albumentations as A

def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

class DataGenerator(Sequence):
    def __init__(self, base_folder, batch_size=8, transform=None, shuffle=True):
        self.folder_B = os.path.join(base_folder, "B")
        self.folder_A = os.path.join(base_folder, "A")
        self.folder_mask = os.path.join(base_folder, "out")
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle

        self.image_files = sorted(os.listdir(self.folder_B))
        self.indexes = np.arange(len(self.image_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.image_files) // self.batch_size

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        X = []
        y = []

        for index in batch_indexes:
            img_B = cv2.imread(os.path.join(self.folder_B, self.image_files[index]))
            img_A = cv2.imread(os.path.join(self.folder_A, self.image_files[index]))
            mask = cv2.imread(os.path.join(self.folder_mask, self.image_files[index]), cv2.IMREAD_GRAYSCALE)

            if self.transform:
                augmented = self.transform(image=img_B, image_after=img_A, mask=mask)
                img_B = augmented['image']
                img_A = augmented['image_after']
                mask = augmented['mask']

            img_B = img_B.astype(np.float32) / 255.0
            img_A = img_A.astype(np.float32) / 255.0
            mask = (mask > 127).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)

            combined_input = np.concatenate([img_B, img_A], axis=-1)
            X.append(combined_input)
            y.append(mask)

        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def create_unet(input_shape=(256,256,6)):
    inputs = Input(input_shape)

    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(2)(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(2)(conv2)

    bottleneck = Conv2D(256, 3, activation='relu', padding='same')(pool2)

    up1 = UpSampling2D(2)(bottleneck)
    concat1 = concatenate([conv2, up1])
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(concat1)

    up2 = UpSampling2D(2)(conv3)
    concat2 = concatenate([conv1, up2])
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(concat2)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv4)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy'])
    return model

def test_with_own_images(path_img1, path_img2, model, img_size=(256,256)):
    img1 = cv2.imread(path_img1)
    img2 = cv2.imread(path_img2)

    img1_resized = cv2.resize(img1, img_size) / 255.0
    img2_resized = cv2.resize(img2, img_size) / 255.0

    input_img = np.concatenate([img1_resized, img2_resized], axis=-1)
    input_img = np.expand_dims(input_img, 0)

    pred_mask = model.predict(input_img)[0, :, :, 0]
    binary_mask = (pred_mask > 0.3).astype(np.uint8) * 255

    kernel = np.ones((5,5), np.uint8)
    
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

    save_path = "predicted_change_cleaned.png"
    cv2.imwrite(save_path, cleaned_mask)
    print(f"\n[+] SUCCESS: Cleaned change mask saved locally as '{save_path}'")

    plt.figure(figsize=(15,5))
    
    plt.subplot(1,4,1)
    plt.title("Before Image")
    plt.imshow(cv2.cvtColor((img1_resized * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.title("After Image")
    plt.imshow(cv2.cvtColor((img2_resized * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.title("Raw Prediction")
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.title("Cleaned Output")
    plt.imshow(cleaned_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_folder = "dataset/train"
    val_folder = "dataset/val"
    test_folder = "dataset/test"
    model_path = "best_model.h5"

    print("Checking for pre-trained model...")
    if os.path.exists(model_path):
        print(f"[{model_path}] found! Loading the trained brain...")
        model = load_model(model_path, custom_objects={'combined_loss': combined_loss, 'dice_loss': dice_loss})
    else:
        print("No trained model found. Setting up AI Training Pipeline from scratch...")
        
        train_transform = A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
        ], additional_targets={'image_after': 'image'})

        val_transform = A.Compose([
            A.Resize(256, 256)
        ], additional_targets={'image_after': 'image'})

        train_data = DataGenerator(train_folder, batch_size=8, transform=train_transform)
        val_data = DataGenerator(val_folder, batch_size=8, transform=val_transform, shuffle=False)

        model = create_unet()

        callbacks = [
            ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1),
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
        ]

        print("Training model...")
        model.fit(train_data, validation_data=val_data, epochs=20, callbacks=callbacks)
        print("Training complete! Brain saved.")

    my_img1 = "images/before.png"
    my_img2 = "images/after.png"
    
    if os.path.exists(my_img1) and os.path.exists(my_img2):
        print("\nTesting custom images...")
        test_with_own_images(my_img1, my_img2, model)
    else:
        print(f"Warning: {my_img1} or {my_img2} not found.")