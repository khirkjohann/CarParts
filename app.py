import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import gdown
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import asyncio
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Car Parts Classification",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Car Parts Classification System - Powered by TensorFlow"}
)

# Properly initialize asyncio event loop
async def init_asyncio():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# Model loading with improved error handling
@st.cache_resource(show_spinner=False)
def load_model() -> Optional[tf.keras.Model]:
    try:
        model_path = 'models/best_model.keras'
        if not os.path.exists('models'):
            os.makedirs('models')

        if not os.path.exists(model_path):
            with st.spinner('🔄 Downloading model... Please wait.'):
                model_url = "https://drive.google.com/uc?id=1R-_GlagW4C7qelQWaDgh9xJ_Ym6qXr6V"
                gdown.download(model_url, output=model_path, quiet=True)

        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
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

class CarPartsDetector(VideoTransformerBase):
    def __init__(self, model: tf.keras.Model, confidence_threshold: float = 0.5):
        super().__init__()
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names
        self._lock = asyncio.Lock()

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized / 255.0
        return np.expand_dims(img_normalized, axis=0)

    async def predict(self, img: np.ndarray) -> tuple:
        async with self._lock:
            try:
                prediction = self.model.predict(img, verbose=0)
                predicted_class_idx = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class_idx]
                return self.class_names[predicted_class_idx], confidence, prediction[0]
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                return "ERROR", 0.0, None

    def draw_prediction_on_frame(self, frame: np.ndarray, class_name: str, confidence: float) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{class_name} ({confidence:.1%})"
        cv2.putText(frame, text, (10, 30), font, 0.7, (0, 255, 0), 2)
        return frame

    async def transform(self, frame) -> np.ndarray:
        try:
            img = frame.to_ndarray(format="bgr")
            processed_frame = self.preprocess_image(img)
            class_name, confidence, _ = await self.predict(processed_frame)
            
            if confidence >= self.confidence_threshold:
                img = self.draw_prediction_on_frame(img, class_name, confidence)
            return img
        except Exception as e:
            logger.error(f"Transform error: {str(e)}")
            return frame.to_ndarray(format="bgr")

async def main():
    st.markdown("## 🚗 Car Parts Classification App")
    st.info("Real-time car parts detection using a machine learning model.")

    # Initialize asyncio loop
    loop = await init_asyncio()

    # Load model
    model = load_model()
    if model is None:
        st.error("❌ Failed to load model. Please try again later.")
        return

    # Sidebar controls
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    # WebRTC configuration with error handling
    try:
        ctx = webrtc_streamer(
            key="car-parts-detector",
            video_processor_factory=lambda: CarPartsDetector(model, confidence_threshold),
            media_stream_constraints={
                "video": {"facingMode": "user"},
                "audio": False
            },
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )
    except Exception as e:
        logger.error(f"WebRTC error: {str(e)}")
        st.error(f"⚠️ Error initializing video stream: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
