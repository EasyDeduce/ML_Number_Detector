# 📝 Handwritten Digit Recognizer (Web App)

A lightweight Deep Learning web application deployed on Streamlit Cloud that detects and recognizes handwritten digits from user uploads.

## 🔗 Live Demo
[Click here to view the App](https://mlnumberdetector.streamlit.app/)

## 🚀 Features
* **Real-time OCR:** Upload any image of handwritten numbers.
* **Smart Preprocessing:** Automatically handles:
    * Transparent backgrounds (PNGs)
    * Low-light photos (Adaptive Thresholding)
    * Thin digital lines (Morphological Dilation)
* **Vertical & Horizontal Sorting:** Automatically detects if numbers are a list (top-to-bottom) or a sentence (left-to-right).

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Inference Engine:** TensorFlow (CPU-optimized)
* **Image Processing:** OpenCV (Headless)

## 📦 Local Installation
To run the app on your own machine:

1.  **Clone the repo:**
    ```bash
    git clone <your-repo-url>
    cd <repo-folder>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

> **Note:** This branch contains only the deployment code. For the model training logic and dataset information, switch to the `training-dev` branch.