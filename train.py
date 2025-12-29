import os
import cv2
import numpy as np
import tensorflow as tf
import kagglehub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

IMG_SIZE = (28, 28)

def get_dataset_root():
    print("Checking/Downloading dataset via kagglehub...")
    try:
        path = kagglehub.dataset_download("jcprogjava/handwritten-digits-dataset-not-in-mnist")
        print(f"Dataset detected at: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        if os.path.exists("dataset"):
            print("Using local 'dataset' folder found in current directory.")
            return "dataset"
        raise FileNotFoundError("Could not download dataset and no local 'dataset' folder found.")

def find_digit_folder(base_path, digit_str):
    for root, dirs, files in os.walk(base_path):
        if digit_str in dirs:
            return os.path.join(root, digit_str)
    return None

def preprocess_image(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return None
        
        if img.shape[-1] == 4:
            img = img[:, :, 3]

        else:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if np.mean(img) > 127:
                img = cv2.bitwise_not(img)

        img = cv2.resize(img, IMG_SIZE)
        return img

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def load_data():
    dataset_root = get_dataset_root()
    data = []
    labels = []
    
    print("Scanning directory structure...")
    
    for digit in range(10):
        folder_path = find_digit_folder(dataset_root, str(digit))
        
        if not folder_path:
            print(f"Warning: Could not find folder for digit '{digit}' inside {dataset_root}")
            continue
            
        print(f"Processing digit {digit} found at: {folder_path}")
        
        file_list = os.listdir(folder_path)
        
        if not file_list:
            print(f"Warning: Folder {folder_path} is empty!")
            continue

        for img_name in file_list:
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(folder_path, img_name)
                img = preprocess_image(img_path)
                
                if img is not None:
                    data.append(img)
                    labels.append(digit)

    if len(data) == 0:
        raise ValueError("No valid images were loaded. Check if dataset is corrupt or empty.")

    data = np.array(data) / 255.0  
    data = np.expand_dims(data, axis=-1)  
    labels = np.array(labels)
    
    print(f"Successfully loaded {len(data)} images.")
    return data, labels

if __name__ == "__main__":
    try:
        X, y = load_data()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        exit(1)

    y = to_categorical(y, num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Starting training...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    save_name = 'digit_model.h5'
    model.save(save_name)
    print(f"Model saved successfully as {save_name}")