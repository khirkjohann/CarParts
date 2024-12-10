import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os
import cv2

# Page configuration (kept the same as original)
st.set_page_config(
    page_title="Car Parts Classification",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Car Parts Classification System - Powered by TensorFlow"
    }
)

# Custom CSS (kept the same as original)
st.markdown("""
    <style>
        .main > div {
            padding: 2rem;
            border-radius: 0.5rem;
        }
        .prediction-box {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border: 1px solid #dee2e6;
            margin: 1rem 0;
        }
        .prediction-list {
            list-style-type: none;
            padding: 0;
        }
        .prediction-list li {
            padding: 0.5rem 0;
            border-bottom: 1px solid #eee;
            font-size: 1.1rem;
        }
        .prediction-list li:last-child {
            border-bottom: none;
        }
        .confidence {
            float: right;
            color: #4CAF50;
            font-weight: bold;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
        }
        .css-1v0mbdj.etr89bj1 {
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Model loading with improved error handling
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model_path = 'models/best_model.keras'
        if not os.path.exists('models'):
            os.makedirs('models')

        if not os.path.exists(model_path):
            with st.spinner('🔄 Downloading model... Please wait.'):
                model_url = "https://drive.google.com/uc?id=1R-_GlagW4C7qelQWaDgh9xJ_Ym6qXr6V"
                gdown.download(model_url, output=model_path, quiet=True)

        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

# Class names and descriptions (kept the same as original)
class_names = [
    'AIR COMPRESSOR', 'ALTERNATOR', 'BATTERY', 'BRAKE CALIPER', 'BRAKE PAD',
    'BRAKE ROTOR', 'CAMSHAFT', 'CARBERATOR', 'CLUTCH PLATE', 'COIL SPRING',
    'CRANKSHAFT', 'CYLINDER HEAD', 'DISTRIBUTOR', 'ENGINE BLOCK', 'ENGINE VALVE',
    'FUEL INJECTOR', 'FUSE BOX', 'GAS CAP', 'HEADLIGHTS', 'IDLER ARM',
    'IGNITION COIL', 'INSTRUMENT CLUSTER', 'LEAF SPRING', 'LOWER CONTROL ARM',
    'MUFFLER', 'OIL FILTER', 'OIL PAN', 'OIL PRESSURE SENSOR', 'OVERFLOW TANK',
    'OXYGEN SENSOR', 'PISTON', 'PRESSURE PLATE', 'RADIATOR', 'RADIATOR FAN',
    'RADIATOR HOSE', 'RADIO', 'RIM', 'SHIFT KNOB', 'SIDE MIRROR', 'SPARK PLUG',
    'SPOILER', 'STARTER', 'TAILLIGHTS', 'THERMOSTAT', 'TORQUE CONVERTER',
    'TRANSMISSION', 'VACUUM BRAKE BOOSTER', 'VALVE LIFTER', 'WATER PUMP',
    'WINDOW REGULATOR'
]

class_info = {
    name: f"Detailed information about {name.lower()} and its function in a vehicle." 
    for name in class_names
}

def preprocess_image(img):
    # Convert PIL Image to numpy array and preprocess
    img_array = np.array(img.convert("RGB"))
    img_resized = tf.image.resize(img_array, [224, 224])
    img_normalized = img_resized / 255.0
    return np.expand_dims(img_normalized, axis=0)

def predict(model, img):
    # Predict the car part and get confidence
    prediction = model.predict(img, verbose=0)
    predicted_class_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_idx]
    return class_names[predicted_class_idx], confidence, prediction[0]

def draw_prediction_on_image(img, class_name, confidence):
    # Convert PIL Image to numpy array
    img_array = np.array(img)
    
    # Create a PIL drawing context
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    # Use a default font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw text on the image
    text = f"{class_name} ({confidence:.1%})"
    draw.text((10, 10), text, fill=(0, 255, 0), font=font)
    
    return img

def main():
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #2E7D32;'>🚗 Car Parts Classification</h1>
        <p style='text-align: center; font-size: 1.2em;'>Upload an image or use the live feed to identify car parts</p>
        <hr>
    """, unsafe_allow_html=True)

    model = load_model()

    if model is None:
        st.error("❌ Failed to load model. Please refresh the page.")
        return

    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Input Options")
        option = st.radio("Select Input Method:", ["Upload Image 📁", "Live Feed 📸"])
        
        st.markdown("### Prediction Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
        
        st.markdown("### ℹ️ About")
        st.info("""
            This application uses machine learning to identify various car parts.
            It can recognize 50 different types of automotive components with high accuracy.
        """)

    if option == "Upload Image 📁":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with st.spinner("🔍 Analyzing image..."):
                processed_img = preprocess_image(image)
                class_name, confidence, all_predictions = predict(model, processed_img)
                
                # Filter by confidence threshold
                if confidence >= confidence_threshold:
                    # Draw prediction on image
                    annotated_image = draw_prediction_on_image(image, class_name, confidence)
                    st.image(annotated_image, caption="Prediction", use_column_width=True)
                
                # Original prediction display
                display_results(class_name, confidence, all_predictions)

    else:  # Live Feed option
        st.markdown("### 📸 Live Camera Feed")
        camera_input = st.camera_input("Take a picture", key="camera_input")
        
        if camera_input:
            image = Image.open(camera_input)
            
            with st.spinner("🔍 Analyzing image..."):
                processed_img = preprocess_image(image)
                class_name, confidence, all_predictions = predict(model, processed_img)
                
                # Filter by confidence threshold
                if confidence >= confidence_threshold:
                    # Draw prediction on image
                    annotated_image = draw_prediction_on_image(image, class_name, confidence)
                    st.image(annotated_image, caption="Prediction", use_column_width=True)
                
                # Original prediction display
                display_results(class_name, confidence, all_predictions)

def display_results(class_name, confidence, all_predictions):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="prediction-box">
                <h3>Primary Prediction</h3>
                <h2 style='color: #4CAF50;'>{}</h2>
                <h4>Confidence: {:.1%}</h4>
            </div>
        """.format(class_name, confidence), unsafe_allow_html=True)
        
        st.markdown("### Part Description")
        st.info(class_info.get(class_name, "No description available."))
    
    with col2:
        st.markdown("### Top Predictions")
        predictions_with_names = list(zip(class_names, all_predictions))
        # Filter predictions with confidence > 0 and sort by confidence
        valid_predictions = [(name, prob) for name, prob in predictions_with_names if prob > 0]
        sorted_predictions = sorted(valid_predictions, key=lambda x: x[1], reverse=True)[:5]
        
        # Display predictions as a clean list
        st.markdown('<ul class="prediction-list">', unsafe_allow_html=True)
        for name, prob in sorted_predictions:
            st.markdown(
                f'<li>{name}<span class="confidence">{prob:.1%}</span></li>',
                unsafe_allow_html=True
            )
        st.markdown('</ul>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
