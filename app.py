import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# Page configuration
st.set_page_config(
    page_title="Car Parts Classification",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Car Parts Classification System - Powered by TensorFlow"
    }
)

# Custom CSS
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

# Class names and descriptions
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
    img = img.convert("RGB")
    img = np.array(img)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict(model, img):
    prediction = model.predict(img, verbose=0)
    predicted_class_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_idx]
    return class_names[predicted_class_idx], confidence, prediction[0]

def draw_prediction_on_image(img, class_name, confidence):
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    text = f"{class_name} ({confidence:.1%})"
    draw.text((10, 10), text, fill=(0, 255, 0), font=font)
    
    return img

def main():
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #2E7D32;'>🚗 Car Parts Classification</h1>
        <p style='text-align: center; font-size: 1.2em;'>Real-Time Car Parts Detection</p>
        <hr>
    """, unsafe_allow_html=True)

    model = load_model()

    if model is None:
        st.error("❌ Failed to load model. Please refresh the page.")
        return

    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Detection Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
        frame_delay = st.slider("Detection Frequency", 1, 10, 3, 1, 
                                help="Lower values process more frames but may reduce performance")
        
        st.markdown("### ℹ️ About")
        st.info("""
            Real-time car parts detection using machine learning.
            Adjust confidence threshold to filter predictions.
        """)

    # Continuous camera input
    camera_placeholder = st.empty()
    predictions_placeholder = st.empty()

    # Frame processing counter
    frame_count = 0

    while True:
        # Capture frame from camera
        camera_input = st.camera_input("", key=f"camera_{frame_count}", label_visibility="collapsed")
        
        if camera_input:
            # Process every nth frame
            frame_count += 1
            if frame_count % frame_delay == 0:
                image = Image.open(camera_input)
                
                with st.spinner("🔍 Analyzing..."):
                    processed_img = preprocess_image(image)
                    class_name, confidence, all_predictions = predict(model, processed_img)
                    
                    # Filter by confidence threshold
                    if confidence >= confidence_threshold:
                        # Draw prediction on image
                        annotated_image = draw_prediction_on_image(image, class_name, confidence)
                        camera_placeholder.image(annotated_image, caption="Current Detection")
                        
                        # Prepare and display top predictions
                        predictions_with_names = list(zip(class_names, all_predictions))
                        valid_predictions = [(name, prob) for name, prob in predictions_with_names if prob >= confidence_threshold]
                        sorted_predictions = sorted(valid_predictions, key=lambda x: x[1], reverse=True)[:5]
                        
                        # Display predictions
                        predictions_html = "<ul class='prediction-list'>"
                        for name, prob in sorted_predictions:
                            predictions_html += f'<li>{name}<span class="confidence">{prob:.1%}</span></li>'
                        predictions_html += '</ul>'
                        predictions_placeholder.markdown(predictions_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
