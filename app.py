import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import gdown
import os
import base64
from io import BytesIO
from streamlit.components.v1 import html
import json

st.set_page_config(page_title="Car Parts Classification", layout="wide")

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model_path = 'models/best_model.keras'
        if not os.path.exists('models'):
            os.makedirs('models')
        if not os.path.exists(model_path):
            with st.spinner('Downloading model...'):
                model_url = "https://drive.google.com/uc?id=1R-_GlagW4C7qelQWaDgh9xJ_Ym6qXr6V"
                gdown.download(model_url, output=model_path, quiet=True)
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

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

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict(model, img):
    prediction = model.predict(img, verbose=0)
    predicted_class_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_idx]
    return class_names[predicted_class_idx], confidence, prediction[0]

def get_webcam_html():
    return """
        <div>
            <video id="video" width="640" height="480" autoplay style="border: 1px solid gray;"></video>
            <canvas id="canvas" width="640" height="480" style="display: none;"></canvas>
        </div>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const context = canvas.getContext('2d');
            let streaming = true;

            navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'environment',
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                } 
            })
            .then(function(stream) {
                video.srcObject = stream;
                video.play();
            })
            .catch(function(err) {
                console.error("Error accessing webcam:", err);
            });

            function captureFrame() {
                if (!streaming) return;
                
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                
                window.parent.postMessage({
                    type: 'webcam-frame',
                    data: imageData
                }, '*');
            }

            // Capture frame every 200ms
            setInterval(captureFrame, 200);

            // Clean up
            window.addEventListener('beforeunload', function() {
                streaming = false;
                if (video.srcObject) {
                    video.srcObject.getTracks().forEach(track => track.stop());
                }
            });
        </script>
    """

def process_webcam_frame(frame_data):
    # Convert base64 to image
    try:
        b64_string = frame_data.split(',')[1]
        img_bytes = base64.b64decode(b64_string)
        img_array = np.array(Image.open(BytesIO(img_bytes)))
        
        # Process and predict
        processed_img = preprocess_image(img_array)
        class_name, confidence, _ = predict(model, processed_img)
        
        return class_name, confidence
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
        return None, None

if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

def main():
    st.title("Car Parts Classification")
    
    global model
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please refresh the page.")
        return

    option = st.sidebar.radio("Select Input Method:", ["Upload Image", "Live Webcam"])

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload a car part image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            img_array = np.array(image)
            processed_img = preprocess_image(img_array)
            
            with st.spinner("Analyzing image..."):
                class_name, confidence, all_predictions = predict(model, processed_img)
            
            st.subheader("Prediction Results")
            st.markdown(f"**Predicted Class:** {class_name}")
            st.markdown(f"**Confidence:** {confidence:.1%}")

    else:
        st.write("Live Webcam Classification")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            html(get_webcam_html(), height=500)
        
        with col2:
            result_placeholder = st.empty()
            confidence_placeholder = st.empty()
            
            def handle_frame(frame_data):
                class_name, confidence = process_webcam_frame(frame_data)
                if class_name and confidence:
                    result_placeholder.markdown(f"**Detected:** {class_name}")
                    confidence_placeholder.markdown(f"**Confidence:** {confidence:.1%}")
            
            st.markdown("### Live Predictions")
            st.markdown("Point your camera at a car part to get real-time predictions.")

if __name__ == "__main__":
    main()
