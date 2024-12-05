import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import gdown
import os
import time

# Page config
st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .upload-text {
        text-align: center;
        padding: 2rem;
        border: 2px dashed #cccccc;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Model loading with progress tracking
@st.cache_resource(show_spinner=False)
def load_detection_model():
    try:
        model_path = 'models/potato_disease_model.keras'
        if not os.path.exists('models'):
            os.makedirs('models')
            
        if not os.path.exists(model_path):
            with st.spinner('Downloading model... This might take a while...'):
                model_url = "https://drive.google.com/file/d/1XuvYZIPSs2LvzohWrza1bWaOMSxu23lr/view?usp=sharing"
                gdown.download(model_url, model_path, quiet=False)
        
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Disease classes with descriptions
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

class_info = {
    'Potato___Early_blight': 'A fungal disease that causes dark spots with concentric rings on potato leaves, typically affecting older leaves first.',
    'Potato___Late_blight': 'A devastating water mold infection causing dark, water-soaked spots on leaves that can quickly destroy entire plants.',
    'Potato___healthy': 'Normal, healthy potato leaves showing no signs of disease.'
}

def predict_image(model, img_array):
    """Predict disease class for preprocessed image array"""
    prediction = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class_idx]
    return class_names[predicted_class_idx], confidence, prediction[0]

def process_image(img):
    """Process image to model input format with error handling"""
    try:
        if len(img.shape) == 2:  # Convert grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = tf.image.resize(img, [224, 224])
        img = img / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def main():
    st.title("🥔 Potato Leaf Disease Detection")
    st.subheader("Upload a potato leaf image to detect diseases")

    # Load model with loading indicator
    with st.spinner("Loading model..."):
        model = load_detection_model()

    if model is None:
        st.error("Failed to load model. Please refresh the page to try again.")
        st.stop()

    # File uploader with drag & drop
    uploaded_file = st.file_uploader(
        "Drag and drop a potato leaf image here",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        try:
            # Create two columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Display original image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

            with col2:
                # Process and predict
                start_time = time.time()
                img_array = np.array(image)
                processed_img = process_image(img_array)
                
                if processed_img is not None:
                    with st.spinner("Analyzing leaf..."):
                        class_name, confidence, all_predictions = predict_image(model, processed_img)
                    
                    # Display prediction results
                    st.subheader("Detection Results")
                    
                    # Display main prediction with colored box
                    prediction_color = {
                        'Potato___healthy': 'success',
                        'Potato___Early_blight': 'warning',
                        'Potato___Late_blight': 'error'
                    }
                    
                    # Clean up class name for display
                    display_name = class_name.replace('Potato___', '').replace('_', ' ').title()
                    
                    st.markdown(f"### Detected Condition: {display_name}")
                    st.markdown(f"**Confidence:** {confidence:.1%}")
                    
                    # Display description
                    st.markdown(f"**Description:**")
                    st.markdown(class_info[class_name])
                    
                    # Show all probabilities
                    st.markdown("### Detailed Analysis")
                    for i, name in enumerate(class_names):
                        prob = all_predictions[i]
                        display_name = name.replace('Potato___', '').replace('_', ' ').title()
                        st.progress(float(prob))
                        st.caption(f"{display_name}: {prob:.1%}")
                    
                    # Display processing time
                    processing_time = time.time() - start_time
                    st.caption(f"Processing time: {processing_time:.2f} seconds")
                    
                    # Add recommendations based on detection
                    if class_name != 'Potato___healthy':
                        st.markdown("### Recommendations")
                        if class_name == 'Potato___Early_blight':
                            st.warning("""
                            - Remove and destroy infected leaves
                            - Apply appropriate fungicides
                            - Ensure proper plant spacing for good air circulation
                            - Water at the base of plants to keep leaves dry
                            """)
                        else:  # Late Blight
                            st.error("""
                            - Immediately isolate infected plants
                            - Apply copper-based fungicides
                            - Improve drainage in the field
                            - Monitor weather conditions for high humidity
                            - Consider removing severely infected plants
                            """)

        except Exception as e:
            st.error(f"Error analyzing image: {str(e)}")
            st.info("Please try uploading a different image")

    # Sidebar information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This application uses deep learning to detect diseases in potato leaves. 
        It can identify:
        - Early Blight
        - Late Blight
        - Healthy Leaves
        
        ### How to use:
        1. Upload an image of a potato leaf
        2. Wait for the analysis
        3. View the detection results and recommendations
        
        ### Tips for best results:
        - Use clear, well-lit images
        - Ensure the leaf is the main focus
        - Avoid blurry or dark images
        """)
        
        # Add expandable section for disease information
        with st.expander("Learn About Potato Diseases"):
            for disease in class_names:
                display_name = disease.replace('Potato___', '').replace('_', ' ').title()
                st.markdown(f"**{display_name}**")
                st.markdown(class_info[disease])
                st.markdown("---")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Created with ❤️ by [Your Name] | 
            <a href="https://github.com/yourusername/potato-disease-detection" target="_blank">GitHub</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
