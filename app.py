import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

np.set_printoptions(suppress=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('digit_model.h5')

model = load_model()

def safe_preprocess(image):
    h, w = image.shape[:2]
    target_height = 300
    aspect_ratio = w / h
    target_width = int(target_height * aspect_ratio)
    
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    
    if resized.shape[-1] == 4:
        gray = resized[:, :, 3] # Alpha channel
        if np.mean(gray) > 250:
             gray = cv2.cvtColor(resized, cv2.COLOR_BGRA2GRAY)
             if np.mean(gray) > 127: gray = cv2.bitwise_not(gray)
    else:
        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized
            
        corners = [gray[0,0], gray[0,-1], gray[-1,0], gray[-1,-1]]
        if np.mean(corners) > 127:
            gray = cv2.bitwise_not(gray)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    return thresh, resized

def prepare_digit_for_model(roi):
    rows, cols = roi.shape
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        
    roi = cv2.resize(roi, (cols, rows))
    
    padded_roi = np.zeros((28, 28), dtype=np.uint8)
    pad_top = (28 - rows) // 2
    pad_left = (28 - cols) // 2
    padded_roi[pad_top:pad_top+rows, pad_left:pad_left+cols] = roi
    
    roi_norm = padded_roi.astype('float32') / 255.0
    roi_norm = np.expand_dims(roi_norm, axis=0)
    roi_norm = np.expand_dims(roi_norm, axis=-1)
    
    return roi_norm

st.title("📝 Handwritten Digit Recognizer")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    st.image(image, caption='Original Upload', width=200)
    
    if st.button('Detect Digits'):
        thresh, resized_img = safe_preprocess(image)
        
        if len(resized_img.shape) == 2:
            output_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
        elif resized_img.shape[2] == 4:
            output_img = cv2.cvtColor(resized_img, cv2.COLOR_BGRA2BGR)
        else:
            output_img = resized_img.copy()

        with st.expander("Debug: View Processed Input"):
            st.write("Model sees this (must be White Text on Black BG):")
            st.image(thresh, width=200, clamp=True)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        if len(contours) > 0:
            (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b:b[1][0]))
        
        detected_digits = []
        found_valid_digit = False

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            
            if w < 5 or h < 10: 
                continue
                
            img_h, img_w = thresh.shape
            if w > 0.9 * img_w and h > 0.9 * img_h:
                continue

            found_valid_digit = True
            roi = thresh[y:y+h, x:x+w]
            
            model_input = prepare_digit_for_model(roi)
            
            prediction = model.predict(model_input, verbose=0)
            digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            detected_digits.append(str(digit))
            
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output_img, str(digit), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if found_valid_digit:
            st.image(output_img, caption='Predictions', use_container_width=True)
            st.success(f"Result: {''.join(detected_digits)}")
        else:
            st.warning("No digits found. The image might be blank or the digit is too faint.")