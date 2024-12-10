import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os
import cv2
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disease management recommendations
disease_recommendations = {
    'Potato___Early_blight': {
        'Prevention': [
            "Maintain proper plant spacing for good air circulation",
            "Water at the base of plants to keep leaves dry",
            "Remove and destroy infected plant debris",
            "Practice crop rotation with non-host crops",
            "Use disease-free seed potatoes"
        ],
        'Treatment': [
            "Apply approved fungicides at first sign of disease",
            "Remove heavily infected leaves",
            "Increase plant spacing to improve air flow",
            "Avoid overhead irrigation",
            "Monitor and maintain proper soil fertility"
        ]
    },
    'Potato___Late_blight': {
        'Prevention': [
            "Plant resistant varieties when available",
            "Improve drainage and air circulation",
            "Monitor weather conditions for disease-favorable environments",
            "Use certified disease-free seed potatoes",
            "Destroy volunteer potato plants and nightshade weeds"
        ],
        'Treatment': [
            "Apply protective fungicides before disease onset",
            "Remove and destroy infected plants immediately",
            "Harvest tubers early if disease is present",
            "Store tubers in cool, dry conditions",
            "Monitor nearby plants for signs of infection"
        ]
    },
    'Potato___healthy': {
        'Maintenance': [
            "Continue regular monitoring",
            "Maintain proper irrigation schedule",
            "Follow recommended fertilization program",
            "Practice good field hygiene",
            "Monitor for early signs of pest or disease problems"
        ]
    }
}

# Model loading
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model_path = 'models/best_model.keras'
        if not os.path.exists('models'):
            os.makedirs('models')

        if not os.path.exists(model_path):
            with st.spinner('Downloading model... Please wait.'):
                model_url = "https://drive.google.com/uc?id=1oTvdOheGhxvxkE4cx3K4nYhTXJB16lpG"
                gdown.download(model_url, output=model_path, quiet=True)

        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

class_info = {
    'Potato___Early_blight': 'A fungal disease that causes dark spots with concentric rings on potato leaves, typically affecting older leaves first.',
    'Potato___Late_blight': 'A devastating water mold infection causing dark, water-soaked spots on leaves that can quickly destroy entire plants.',
    'Potato___healthy': 'Normal, healthy potato leaves showing no signs of disease.'
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

def display_recommendations(class_name):
    if class_name == 'Potato___healthy':
        st.markdown("### Maintenance Recommendations")
        for tip in disease_recommendations[class_name]['Maintenance']:
            st.markdown(f"• {tip}")
    else:
        st.markdown("### Prevention Recommendations")
        for tip in disease_recommendations[class_name]['Prevention']:
            st.markdown(f"• {tip}")
        
        st.markdown("### Treatment Recommendations")
        for tip in disease_recommendations[class_name]['Treatment']:
            st.markdown(f"• {tip}")

def real_time_detection():
    st.markdown("### Real-time Detection")
    camera_placeholder = st.empty()
    stop_button = st.button("Stop Detection")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break
            
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Process and predict
        processed_img = preprocess_image(pil_image)
        class_name, confidence, all_predictions = predict(model, processed_img)
        
        # Display frame with overlay
        frame_with_text = frame_rgb.copy()
        cv2.putText(frame_with_text, 
                   f"{class_name}: {confidence:.1%}", 
                   (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, 
                   (0, 255, 0) if class_name == 'Potato___healthy' else (0, 0, 255), 
                   2)
        
        camera_placeholder.image(frame_with_text)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Current Detection:** {class_name}")
            st.markdown(f"**Confidence:** {confidence:.1%}")
            
        with col2:
            st.markdown("### Description")
            st.markdown(class_info.get(class_name, "No description available."))
        
        # Display recommendations
        display_recommendations(class_name)
        
        time.sleep(0.1)  # Add small delay to prevent overwhelming the system
    
    cap.release()

def main():
    st.title("Potato Leaf Disease Detection")
    
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please refresh the page.")
        return
    
    # Sidebar options
    option = st.sidebar.radio("Select Input Method:", ["Upload Image", "Live Feed", "Real-time Detection"])
    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            processed_img = preprocess_image(image)
            
            with st.spinner("Analyzing image..."):
                class_name, confidence, all_predictions = predict(model, processed_img)
            
            st.subheader("Prediction Results")
            st.markdown(f"**Predicted Class:** {class_name}")
            st.markdown(f"**Confidence:** {confidence:.1%}")
            
            # Display detailed probabilities
            st.markdown("### Class Probabilities")
            predictions_with_names = list(zip(class_names, all_predictions))
            for name, prob in sorted(predictions_with_names, key=lambda x: x[1], reverse=True):
                if prob > 0:
                    st.progress(float(prob))
                    st.caption(f"{name}: {prob:.1%}")
            
            # Description and Recommendations
            st.markdown("### Description")
            st.markdown(class_info.get(class_name, "No description available."))
            
            display_recommendations(class_name)
            
    elif option == "Live Feed":
        camera_input = st.camera_input("Take a picture")
        
        if camera_input is not None:
            image = Image.open(camera_input)
            st.image(image, caption="Captured Image", use_column_width=True)
            
            processed_img = preprocess_image(image)
            
            with st.spinner("Analyzing image..."):
                class_name, confidence, all_predictions = predict(model, processed_img)
            
            st.subheader("Prediction Results")
            st.markdown(f"**Predicted Class:** {class_name}")
            st.markdown(f"**Confidence:** {confidence:.1%}")
            
            # Display detailed probabilities and recommendations
            st.markdown("### Class Probabilities")
            predictions_with_names = list(zip(class_names, all_predictions))
            for name, prob in sorted(predictions_with_names, key=lambda x: x[1], reverse=True):
                if prob > 0:
                    st.progress(float(prob))
                    st.caption(f"{name}: {prob:.1%}")
            
            st.markdown("### Description")
            st.markdown(class_info.get(class_name, "No description available."))
            
            display_recommendations(class_name)
            
    elif option == "Real-time Detection":
        real_time_detection()

if __name__ == "__main__":
    main()
