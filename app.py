import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K
from PIL import Image
import cv2
import numpy as np
import gdown
import os
from tensorflow.keras.preprocessing import image
import io

# Page config
st.set_page_config(
    page_title="Car Parts Detection",
    page_icon="🚗",
    layout="wide"
)

# Custom F1 score metric
def F1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# Model loading with error handling
@st.cache_resource
def load_detection_model():
    try:
        model_path = 'car_parts_model.keras'
        if not os.path.exists('models'):
            os.makedirs('models')
            
        if not os.path.exists(model_path):
            with st.spinner('Downloading model... This might take a while...'):
                # Get the model URL from Streamlit secrets
                model_url = "https://drive.google.com/file/d/1u1kzAKwzG6DUH5-4YAbsgqpjZKHEWgjr/view?usp=sharing"
                gdown.download(model_url, model_path, quiet=False)
        
        return tf.keras.models.load_model(model_path, custom_objects={'F1_score': F1_score})
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load class names
class_names = [
    'AIR COMPRESSOR', 'ALTERNATOR', 'BATTERY', 'BRAKE CALIPER', 'BRAKE PAD', 'BRAKE ROTOR', 
    'CAMSHAFT', 'CARBERATOR', 'CLUTCH PLATE', 'COIL SPRING', 'CRANKSHAFT', 'CYLINDER HEAD', 
    'DISTRIBUTOR', 'ENGINE BLOCK', 'ENGINE VALVE', 'FUEL INJECTOR', 'FUSE BOX', 'GAS CAP', 
    'HEADLIGHTS', 'IDLER ARM', 'IGNITION COIL', 'INSTRUMENT CLUSTER', 'LEAF SPRING', 
    'LOWER CONTROL ARM', 'MUFFLER', 'OIL FILTER', 'OIL PAN', 'OIL PRESSURE SENSOR', 
    'OVERFLOW TANK', 'OXYGEN SENSOR', 'PISTON', 'PRESSURE PLATE', 'RADIATOR', 'RADIATOR FAN', 
    'RADIATOR HOSE', 'RADIO', 'RIM', 'SHIFT KNOB', 'SIDE MIRROR', 'SPARK PLUG', 'SPOILER', 
    'STARTER', 'TAILLIGHTS', 'THERMOSTAT', 'TORQUE CONVERTER', 'TRANSMISSION', 
    'VACUUM BRAKE BOOSTER', 'VALVE LIFTER', 'WATER PUMP', 'WINDOW REGULATOR'
]

def predict_image(img_array):
    """Predict class for preprocessed image array"""
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class_idx]
    return class_names[predicted_class_idx], confidence

def process_image(img):
    """Process image to model input format"""
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Main app
st.title("🚗 Car Parts Detection System")

# Load model
model = load_detection_model()

if model is None:
    st.error("Failed to load model. Please try again later.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process and predict
        img_array = np.array(image)
        processed_img = process_image(img_array)
        class_name, confidence = predict_image(processed_img)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"Detected Part: {class_name}")
        with col2:
            st.info(f"Confidence: {confidence:.2%}")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Add information about the app
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info("""
This application uses a deep learning model to detect various car parts.
It can identify 50 different types of car parts from uploaded images.
""")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Created with ❤️ by [Your Name]</p>
    </div>
    """,
    unsafe_allow_html=True
)
