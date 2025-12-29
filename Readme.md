# 🧠 Model Training & Architecture

This branch contains the source code used to train the Convolutional Neural Network (CNN) for the Handwritten Digit Recognizer.

## 📂 Dataset
We use the **[Handwritten Digits Dataset (Not in MNIST)](https://www.kaggle.com/datasets/jcprogjava/handwritten-digits-dataset-not-in-mnist)**.
* **Data Source:** Kaggle
* **Automation:** The script uses `kagglehub` to automatically download and cache the data. You do *not* need to manually download files.

## 🧠 Model Architecture
We use a custom CNN optimized for spatial feature extraction:
* **Input:** 28x28 Grayscale images (inverted to White-on-Black).
* **Layers:**
    * 2x `Conv2D` (Feature extraction) + `MaxPooling2D`
    * `Flatten` -> `Dense` (128 units)
    * `Dropout` (0.5) to prevent overfitting
    * `Softmax` Output (10 classes)

## ⚡ How to Train

1.  **Install Training Dependencies:**
    (Note: These differ from the deployment dependencies)
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Script:**
    ```bash
    python train.py
    ```

3.  **Output:**
    The script will generate a file named `digit_model.h5`.
    * *Accuracy:* ~98%
    * *Loss:* Categorical Crossentropy

## ⚙️ Requirements
See `requirements_train.txt` for the full list, including `tensorflow` (GPU supported) and `kagglehub`.

