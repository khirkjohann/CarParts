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

def get_webcam_html():
    return """
        <div style="max-width: 414px; margin: 0 auto;">
            <video id="video" width="414" height="736" autoplay style="border: 1px solid gray; border-radius: 20px;"></video>
            <canvas id="canvas" width="414" height="736" style="display: none;"></canvas>
            <button id="capture" style="
                width: 70px;
                height: 70px;
                border-radius: 35px;
                background-color: white;
                border: 3px solid gray;
                position: relative;
                margin: 20px auto;
                display: block;
                cursor: pointer;">
            </button>
        </div>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const captureButton = document.getElementById('capture');
            const context = canvas.getContext('2d');
            let streaming = true;

            navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'environment',
                    width: { ideal: 414 },
                    height: { ideal: 736 }
                } 
            })
            .then(function(stream) {
                video.srcObject = stream;
                video.play();
            })
            .catch(function(err) {
                console.error("Error accessing webcam:", err);
            });

            captureButton.addEventListener('click', function() {
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                window.parent.postMessage({
                    type: 'webcam-capture',
                    data: imageData
                }, '*');
            });

            // Clean up
            window.addEventListener('beforeunload', function() {
                streaming = false;
                if (video.srcObject) {
                    video.srcObject.getTracks().forEach(track => track.stop());
                }
            });
        </script>
    """

def main():
    st.title("Car Parts Classification")
    
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please refresh the page.")
        return

    option = st.sidebar.radio("Select Input Method:", ["Upload Image", "Live Webcam"])

    if option == "Upload Image":
        # [Previous upload code remains the same]
        pass

    else:
        st.write("Point camera at a car part and tap the capture button")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            html(get_webcam_html(), height=850)
        
        with col2:
            result_container = st.container()
            
            if 'captured_image' in st.session_state:
                with result_container:
                    st.image(st.session_state.captured_image, caption="Captured Image")
                    processed_img = preprocess_image(np.array(st.session_state.captured_image))
                    class_name, confidence, all_predictions = predict(model, processed_img)
                    
                    st.markdown(f"**Top Prediction:** {class_name}")
                    st.markdown(f"**Confidence:** {confidence:.1%}")
                    
                    st.markdown("### All Predictions")
                    # Sort predictions by confidence
                    predictions = [(name, float(pred)) for name, pred in zip(class_names, all_predictions)]
                    predictions.sort(key=lambda x: x[1], reverse=True)
                    
                    # Show top 10 predictions
                    for name, conf in predictions[:10]:
                        st.progress(conf)
                        st.caption(f"{name}: {conf:.1%}")

        # Handle webcam capture
        if st.session_state.get('webcam_frame'):
            try:
                b64_string = st.session_state.webcam_frame.split(',')[1]
                img_bytes = base64.b64decode(b64_string)
                captured_image = Image.open(BytesIO(img_bytes))
                st.session_state.captured_image = captured_image
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error processing capture: {str(e)}")

if __name__ == "__main__":
    main()
