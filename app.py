import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import gdown
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import asyncio

# Page configuration
st.set_page_config(
    page_title="Car Parts Classification",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Car Parts Classification System - Powered by TensorFlow"}
)

# Set up an asyncio event loop explicitly
if not asyncio.get_event_loop_policy().get_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

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

# Class names
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

# Video transformer
class CarPartsDetector(VideoTransformerBase):
    def __init__(self, model, confidence_threshold=0.5):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names

    def preprocess_image(self, img):
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized / 255.0
        return np.expand_dims(img_normalized, axis=0)

    def predict(self, img):
        prediction = self.model.predict(img, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]
        return self.class_names[predicted_class_idx], confidence, prediction[0]

    def draw_prediction_on_frame(self, frame, class_name, confidence):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{class_name} ({confidence:.1%})"
        cv2.putText(frame, text, (10, 30), font, 0.7, (0, 255, 0), 2)
        return frame

    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr")
            processed_frame = self.preprocess_image(img)
            class_name, confidence, _ = self.predict(processed_frame)
            if confidence >= self.confidence_threshold:
                img = self.draw_prediction_on_frame(img, class_name, confidence)
            return img
        except Exception as e:
            st.error(f"Error in video processing: {e}")
            return frame.to_ndarray(format="bgr")

def main():
    st.markdown("## 🚗 Car Parts Classification App")
    st.info("Real-time car parts detection using a machine learning model.")

    model = load_model()
    if model is None:
        st.error("❌ Failed to load model. Please try again later.")
        return

    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    ctx = webrtc_streamer(
        key="car-parts-detector",
        video_processor_factory=lambda: CarPartsDetector(model, confidence_threshold),
        media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
        async_processing=True,
    )

if __name__ == "__main__":
    main()
