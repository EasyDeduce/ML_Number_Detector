import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# --- CONFIGURATION ---
# Disable scientific notation for clearer output
np.set_printoptions(suppress=True)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('digit_model.h5')

model = load_model()

def preprocess_image(image):
    """
    Preprocesses the image to match Model Input (White Digit / Black BG).
    Handles both:
    1. Transparent PNGs (Dataset style)
    2. Camera Photos (Paper style)
    """
    # Check if image has 4 channels (Transparency/Alpha)
    if image.shape[-1] == 4:
        # Extract Alpha channel: 0=Transparent (BG), 255=Opaque (Digit)
        # The dataset has black digits (opaque) on transparent BG. 
        # The alpha channel essentially gives us the "mask" of the digit.
        gray = image[:, :, 3]
        
        # Apply a simple threshold to clean it up
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    else:
        # Standard RGB/BGR Image (e.g., from Phone Camera)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Thresholding: 
        # We use THRESH_BINARY_INV because usually we have Dark Digits on Light Paper.
        # We want to flip that to Light Digits on Dark Background.
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return thresh

def sort_contours(cnts):
    # Sort contours left-to-right
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b:b[1][0]))
    return cnts

# --- UI LAYOUT ---
st.title("📝 Handwritten Digit Recognizer")
st.markdown("Upload an image (Phone photo or PNG) to detect digits.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read file buffer as byte array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # LOAD AS 'UNCHANGED' TO KEEP TRANSPARENCY IF PRESENT
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    # Display original image
    # Note: Streamlit handles transparency well in display
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Detect Digits'):
        # 1. Preprocess (Get White Digit on Black BG)
        thresh = preprocess_image(image)
        
        # Debug: Show what the computer sees (Optional, good for troubleshooting)
        with st.expander("See what the model sees"):
             st.image(thresh, caption='Processed Input (White on Black)', clamp=True)
        
        # 2. Find Contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_digits = []
        digit_rects = [] # Store coords to draw boxes later
        
        if len(contours) > 0:
            contours = sort_contours(contours)
            
            # If the image was transparent, we need a standard BGR copy to draw colored boxes on
            if len(image.shape) == 3: 
                output_img = image.copy()
            else:
                # Convert BGRA to BGR for drawing
                output_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                
                # Filter noise: Ignore very small dots
                if w > 5 and h > 15:
                    # ROI (Region of Interest)
                    roi = thresh[y:y+h, x:x+w]
                    
                    # Resize logic (preserve aspect ratio)
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
                    
                    # Pad to 28x28
                    padded_roi = np.zeros((28, 28), dtype=np.uint8)
                    pad_top = (28 - rows) // 2
                    pad_left = (28 - cols) // 2
                    padded_roi[pad_top:pad_top+rows, pad_left:pad_left+cols] = roi
                    
                    # Normalize
                    roi_norm = padded_roi.astype('float32') / 255.0
                    roi_norm = np.expand_dims(roi_norm, axis=0) 
                    roi_norm = np.expand_dims(roi_norm, axis=-1)
                    
                    # Predict
                    prediction = model.predict(roi_norm, verbose=0)
                    digit = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    # Only accept if confidence is decent (optional filter)
                    if confidence > 0.4:
                        detected_digits.append(str(digit))
                        
                        # Draw box and text
                        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(output_img, str(digit), (x, y - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Show Result
            st.image(output_img, caption='Detected Digits', use_container_width=True)
            
            if detected_digits:
                st.success(f"Predicted Number: {''.join(detected_digits)}")
            else:
                st.warning("Contours found, but they looked like noise. Try a clearer image.")
            
        else:
            st.error("No digits detected. Try writing darker or using a plain background.")