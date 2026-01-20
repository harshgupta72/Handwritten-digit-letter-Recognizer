
"""
Handwritten Digit & Letter Recognizer - Streamlit Web Application
Interactive web app for digit and letter recognition
"""

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import string

# Page configuration
st.set_page_config(
    page_title="Handwritten Recognizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        /* Global Reset & Font */
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        
        /* Background Gradient for the whole app - Light Theme */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #333333;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e0e0e0;
        }
        
        /* Main Title Styling with Gradient Text */
        .main-title {
            text-align: center;
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #4b6cb7, #182848);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
            padding-bottom: 10px;
        }
        
        .subtitle {
            text-align: center;
            color: #555555;
            font-size: 1.1rem;
            margin-bottom: 40px;
        }
        
        /* Card Container Styling */
        .card-container {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.5);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            height: 100%;
        }
        
        /* Metric Styling */
        div[data-testid="stMetricValue"] {
            font-size: 2rem;
            color: #2c3e50; /* Dark Blue */
        }
        
        div[data-testid="stMetricLabel"] {
            color: #555555;
        }
        
        /* Canvas Styling */
        .canvas-container {
            border: 2px dashed #a0a0a0;
            border-radius: 15px;
            padding: 10px;
            background: #ffffff;
            display: flex;
            justify-content: center;
        }
        
        /* Buttons */
        div.stButton > button {
            background: linear-gradient(90deg, #4b6cb7, #182848);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(24, 40, 72, 0.3);
        }
        
        /* Custom Progress Bar */
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #4b6cb7, #182848);
        }
        
        /* Headers inside cards */
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
        }
        
        /* Char Grid Item */
        .char-item {
            background: #ffffff;
            border-radius: 10px;
            padding: 5px;
            text-align: center;
            margin-bottom: 10px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
    </style>
    """, unsafe_allow_html=True)



@st.cache_resource
def load_model(model_type):
    """Load the trained model (cached)"""
    try:
        if model_type == 'digit':
            path = 'models/final_model.h5'
            if not tf.io.gfile.exists(path):
                 st.error(f"Model not found at {path}. Please run train_model.py first.")
                 return None
        else:
            path = 'models/text_recognizer_model.h5'
            if not tf.io.gfile.exists(path):
                 st.error(f"Model not found at {path}. Please run train_text_model.py first.")
                 return None
                 
        model = keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading {model_type} model: {e}")
        return None

def segment_characters(image):
    """Segment characters from an image containing a word/sentence"""
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Binary thresholding (assuming white text on black background from canvas)
    # If uploaded image (black text on white), invert first
    if np.mean(gray) > 127: # Light background
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    else: # Dark background
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    if not bounding_boxes:
        return []

    # Sort by x coordinate
    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes),
                                            key=lambda b: b[1][0]))

    segmented_images = []
    
    # Process each contour
    for c, box in zip(contours, bounding_boxes):
        x, y, w, h = box
        
        # Filter small noise
        if w < 5 or h < 5:
            continue
            
        # Extract ROI
        roi = thresh[y:y+h, x:x+w]
        
        # Resize to 20x20 while maintaining aspect ratio (like MNIST)
        # Create a blank 20x20 image
        canvas_20 = np.zeros((20, 20), dtype=np.uint8)
        
        # Calculate scaling factor
        scale = min(20/w, 20/h)
        nw = int(w * scale)
        nh = int(h * scale)
        
        # Resize ROI
        roi_resized = cv2.resize(roi, (nw, nh))
        
        # Center on 20x20 canvas
        start_x = (20 - nw) // 2
        start_y = (20 - nh) // 2
        canvas_20[start_y:start_y+nh, start_x:start_x+nw] = roi_resized
        
        # Pad to 28x28 (add 4 pixels border)
        final_img = np.pad(canvas_20, ((4,4),(4,4)), "constant", constant_values=0)
        
        segmented_images.append(final_img)
        
    return segmented_images

def preprocess_single_char(img_array):
    """Preprocess a single character image (28x28) for prediction"""
    # Normalize
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape for model
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

def preprocess_image(image):
    """Legacy preprocess function for full image (deprecated/fallback)"""
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Resize to 28x28
    img_array = cv2.resize(img_array, (28, 28))
    
    # Invert if background is white
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # Normalize
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape for model
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

def get_prediction_label(index, model_type):
    if model_type == 'digit':
        return str(index)
    else:
        # 0 -> A, 1 -> B, ...
        return string.ascii_uppercase[index]

def main():
    """Main Streamlit app"""
    inject_custom_css()
    
    st.markdown('<h1 class="main-title">📝 Handwritten Recognizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Powered by Convolutional Neural Networks (CNN)</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        mode = st.radio("Select Recognition Mode", [
            "Digit Recognition (0-9)", 
            "Letter Recognition (A-Z)",
            "Sentence/Word Recognition"
        ], help="Choose what you want the AI to recognize.")
        
        if "Digit" in mode:
            model_type = 'digit'
        elif "Letter" in mode:
            model_type = 'letter'
        else:
            model_type = 'letter' # Use letter model for words by default
        
        st.markdown("---")
        st.info(f"**Current Model:** {model_type.capitalize()} Recognizer")
        
        with st.expander("ℹ️ About this App"):
            st.markdown(f"""
            This smart application uses Deep Learning to recognize your handwriting!
            
            **How it works:**
            1. You draw or upload an image.
            2. The image is processed (resized, normalized).
            3. A CNN model predicts the character.
            
            **Tech Stack:**
            - TensorFlow/Keras
            - OpenCV
            - Streamlit
            """)
    
    # Load model
    model = load_model(model_type)
    if model is None:
        st.stop()
        
    # Main content area
    col1, col2 = st.columns([4, 5], gap="large")
    
    with col1:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown("### 📤 Input Area")
        st.markdown("Draw your input below or upload an image.")
        
        # Option selection
        input_method = st.radio(
            "Input Method:",
            ["Draw on Canvas", "Upload Image"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        image = None
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                f"Upload image (PNG, JPG)",
                type=['png', 'jpg', 'jpeg']
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
        
        else:  # Draw
            # Increase canvas size for words
            is_word_mode = "Word" in mode
            width = 600 if is_word_mode else 300
            height = 200 if is_word_mode else 300
            
            st.markdown(f'<div class="canvas-container">', unsafe_allow_html=True)
            canvas_result = st_canvas(
                fill_color="black",
                stroke_width=10 if is_word_mode else 20,
                stroke_color="white",
                background_color="black",
                width=width,
                height=height,
                drawing_mode="freedraw",
                key=f"canvas_{mode}_{model_type}",
                display_toolbar=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("Tip: Use the trash icon on the canvas toolbar to clear.")
            
            if canvas_result.image_data is not None:
                # Check if canvas is not empty (all black)
                if np.max(canvas_result.image_data) > 0:
                    image = Image.fromarray(canvas_result.image_data.astype('uint8'))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown("### 🎯 Results")
        
        if image is not None:
            if "Word" in mode:
                # Segmentation logic
                with st.spinner("Analyzing text..."):
                    chars = segment_characters(image)
                
                if not chars:
                    st.warning("⚠️ No characters detected. Please draw clearly.")
                else:
                    st.success(f"✅ Detected {len(chars)} characters!")
                    
                    full_word = ""
                    
                    st.markdown("#### Character Breakdown:")
                    # Create a scrolling container or grid for characters
                    cols = st.columns(min(len(chars), 6))
                    
                    for i, char_img in enumerate(chars):
                        # Preprocess & Predict
                        processed = preprocess_single_char(char_img)
                        preds = model.predict(processed, verbose=0)
                        idx = np.argmax(preds)
                        label = get_prediction_label(idx, model_type)
                        conf = float(preds[0][idx] * 100)
                        
                        full_word += label
                        
                        # Display in grid with new styling
                        with cols[i % 6]:
                            st.markdown('<div class="char-item">', unsafe_allow_html=True)
                            st.image(char_img, width=40, clamp=True)
                            st.markdown(f"**{label}**")
                            st.markdown(f"<span style='font-size:0.8em; color:#00d4ff'>{conf:.0f}%</span>", unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.metric("Predicted Text", full_word)
            
            else:
                # Original Single Character Logic
                processed_img = preprocess_image(image)
                
                # Get prediction
                predictions = model.predict(processed_img, verbose=0)
                predicted_index = np.argmax(predictions)
                predicted_label = get_prediction_label(predicted_index, model_type)
                confidence = float(predictions[0][predicted_index] * 100)
                
                # Display Metrics
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Prediction", predicted_label)
                with m2:
                    st.metric("Confidence", f"{confidence:.2f}%")
                
                # Tabs for details
                tab1, tab2 = st.tabs(["📊 Probabilities", "🖼️ Debug View"])
                
                with tab1:
                    # Top 3 predictions
                    st.caption("Top 3 Candidates:")
                    top3_indices = np.argsort(predictions[0])[-3:][::-1]
                    top3_probs = predictions[0][top3_indices] * 100
                    
                    for idx, prob in zip(top3_indices, top3_probs):
                        label = get_prediction_label(idx, model_type)
                        prob_val = float(prob)
                        st.markdown(f"**{label}**: {prob_val:.2f}%")
                        st.progress(float(prob_val / 100))
                    
                    # Full Chart
                    if model_type == 'digit':
                        labels = [str(i) for i in range(10)]
                    else:
                        labels = list(string.ascii_uppercase)
                        
                    chart_data = pd.DataFrame({
                        'Label': labels,
                        'Probability': predictions[0] * 100
                    })
                    st.bar_chart(chart_data.set_index('Label'), height=200)

                with tab2:
                    st.write("Model Input (28x28):")
                    st.image(processed_img.squeeze(), width=100, clamp=True)
            
        else:
            # Placeholder when no input
            st.info("👈 Waiting for input...")
            st.markdown(
                """
                <div style="text-align: center; opacity: 0.5; padding: 20px;">
                    <h1 style="font-size: 5rem;">🖊️</h1>
                    <p>Draw something on the left to see magic happen!</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #555555; font-size: 0.9em; padding: 15px;'>
            <p>Developed by <b style="color: #4b6cb7;">Harsh Gupta</b></p>
            <p style="font-size: 0.8em; opacity: 0.7;">© 2026 All Rights Reserved</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
