import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# Page configuration
st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Class names and descriptions (update as per your dataset)
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

class_info = {
    'Potato___Early_blight': 'A fungal disease that causes dark spots with concentric rings on potato leaves, typically affecting older leaves first.',
    'Potato___Late_blight': 'A devastating water mold infection causing dark, water-soaked spots on leaves that can quickly destroy entire plants.',
    'Potato___healthy': 'Normal, healthy potato leaves showing no signs of disease.'
}
# Image preprocessing
def preprocess_image(img):
    img = img.convert("RGB")  # Convert to RGB if it's not already
    img = np.array(img)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Prediction function
def predict(model, img):
    prediction = model.predict(img, verbose=0)
    predicted_class_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_idx]
    return class_names[predicted_class_idx], confidence, prediction[0]

# Main app
def main():
    st.title("Car Parts Classification")
    st.subheader("Upload an image or use the live feed to classify car parts")

    model = load_model()

    if model is None:
        st.error("Failed to load model. Please refresh the page.")
        return

    # Sidebar options
    option = st.sidebar.radio("Select Input Method:", ["Upload Image", "Live Feed"])

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload a car part image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            processed_img = preprocess_image(image)

            with st.spinner("Analyzing image..."):
                class_name, confidence, all_predictions = predict(model, processed_img)

            st.subheader("Prediction Results")
            st.markdown(f"**Predicted Class:** {class_name}")
            st.markdown(f"**Confidence:** {confidence:.1%}")

            # Sort and filter predictions
            predictions_with_names = list(zip(class_names, all_predictions))
            sorted_predictions = sorted(predictions_with_names, key=lambda x: x[1], reverse=True)
            non_zero_predictions = [(name, prob) for name, prob in sorted_predictions if prob > 0]

            # Display detailed probabilities
            st.markdown("### Class Probabilities")
            for name, prob in non_zero_predictions:
                st.progress(float(prob))
                st.caption(f"{name}: {prob:.1%}")

            # Description
            st.markdown("### Part Description")
            st.markdown(class_info.get(class_name, "No description available."))

    elif option == "Live Feed":
        st.write("Click the button below to start the live feed.")

        # Use st.camera_input to access the device's camera
        camera_input = st.camera_input("Take a picture")

        if camera_input is not None:
            # Open the captured image
            image = Image.open(camera_input)
            st.image(image, caption="Captured Image", use_column_width=True)

            # Preprocess the image and get predictions
            processed_img = preprocess_image(image)

            with st.spinner("Analyzing image..."):
                class_name, confidence, all_predictions = predict(model, processed_img)

            st.subheader("Prediction Results")
            st.markdown(f"**Predicted Class:** {class_name}")
            st.markdown(f"**Confidence:** {confidence:.1%}")

            # Sort and filter predictions
            predictions_with_names = list(zip(class_names, all_predictions))
            sorted_predictions = sorted(predictions_with_names, key=lambda x: x[1], reverse=True)
            non_zero_predictions = [(name, prob) for name, prob in sorted_predictions if prob > 0]

            # Display detailed probabilities
            st.markdown("### Class Probabilities")
            for name, prob in non_zero_predictions:
                st.progress(float(prob))
                st.caption(f"{name}: {prob:.1%}")

            # Description
            st.markdown("### Description")
            st.markdown(class_info.get(class_name, "No description available."))

if __name__ == "__main__":
    main()
