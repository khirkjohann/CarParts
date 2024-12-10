import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import gdown
import os

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
    # Resize and normalize the image for model prediction
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    return np.expand_dims(img_normalized, axis=0)

def predict(model, img):
    # Predict the car part and get confidence
    prediction = model.predict(img, verbose=0)
    predicted_class_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_idx]
    return class_names[predicted_class_idx], confidence, prediction[0]

def draw_prediction_on_frame(frame, class_name, confidence):
    # Draw prediction text on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    
    # Top prediction box
    text = f"{class_name} ({confidence:.1%})"
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    cv2.rectangle(frame, (10, 30), (20 + text_width, 50), (0, 255, 0), -1)
    cv2.putText(frame, text, (15, 50), font, font_scale, (0, 0, 0), font_thickness)
    
    return frame

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
        st.markdown("### 🛠️ Detection Options")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
        frame_skip = st.slider("Frame Processing Rate", 1, 10, 3, 1, 
                               help="Lower values process more frames but may reduce performance")
        
        st.markdown("### ℹ️ About")
        st.info("""
            Real-time car parts detection using machine learning.
            Adjust confidence threshold to filter predictions.
        """)

    # Real-time detection section
    st.markdown("### 📸 Real-Time Detection")
    
    # Create placeholders for video stream and sidebar info
    video_placeholder = st.empty()
    top_predictions_placeholder = st.empty()

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Frame counter for processing optimization
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        # Process every nth frame to reduce computation
        frame_count += 1
        if frame_count % frame_skip == 0:
            # Prepare frame for prediction
            processed_frame = preprocess_image(frame)

            # Predict car part
            class_name, confidence, all_predictions = predict(model, processed_frame)

            # Filter predictions by confidence threshold
            if confidence >= confidence_threshold:
                # Draw prediction on frame
                frame = draw_prediction_on_frame(frame, class_name, confidence)

                # Prepare top predictions for sidebar
                predictions_with_names = list(zip(class_names, all_predictions))
                valid_predictions = [(name, prob) for name, prob in predictions_with_names if prob >= confidence_threshold]
                sorted_predictions = sorted(valid_predictions, key=lambda x: x[1], reverse=True)[:5]

                # Display top predictions
                top_predictions_html = "<ul class='prediction-list'>"
                for name, prob in sorted_predictions:
                    top_predictions_html += f'<li>{name}<span class="confidence">{prob:.1%}</span></li>'
                top_predictions_html += '</ul>'
                top_predictions_placeholder.markdown(top_predictions_html, unsafe_allow_html=True)

        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        video_placeholder.image(frame_rgb, channels="RGB")

        # Break the loop if user stops the app
        if not st.session_state.get('run_detection', True):
            break

    # Release the webcam
    cap.release()

if __name__ == "__main__":
    main()
