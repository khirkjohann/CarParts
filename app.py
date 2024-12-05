import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import gdown
import os
import time
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Data structures
@dataclass
class DiseaseInfo:
    display_name: str
    description: str
    recommendations: List[str]
    severity_level: str  # 'success', 'warning', or 'error'

# Constants and configurations
DISEASE_MAPPING: Dict[str, DiseaseInfo] = {
    'Potato___Early_blight': DiseaseInfo(
        display_name='Early Blight',
        description='A fungal disease that causes dark spots with concentric rings on potato leaves, typically affecting older leaves first.',
        recommendations=[
            'Remove and destroy infected leaves',
            'Apply appropriate fungicides',
            'Ensure proper plant spacing for good air circulation',
            'Water at the base of plants to keep leaves dry'
        ],
        severity_level='warning'
    ),
    'Potato___Late_blight': DiseaseInfo(
        display_name='Late Blight',
        description='A devastating water mold infection causing dark, water-soaked spots on leaves that can quickly destroy entire plants.',
        recommendations=[
            'Immediately isolate infected plants',
            'Apply copper-based fungicides',
            'Improve drainage in the field',
            'Monitor weather conditions for high humidity',
            'Consider removing severely infected plants'
        ],
        severity_level='error'
    ),
    'Potato___healthy': DiseaseInfo(
        display_name='Healthy',
        description='Normal, healthy potato leaves showing no signs of disease.',
        recommendations=[],
        severity_level='success'
    )
}

MODEL_CONFIG = {
    'path': 'models/potato_disease_model.keras',
    'url': "https://drive.google.com/uc?id=1NF8DeUe4gy_x6NtZh12-IwD4q8n0RDjI",
    'input_size': (224, 224)
}

# Custom CSS with added video feed styling
CUSTOM_CSS = """
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
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
    }
    .video-container {
        border: 2px solid #cccccc;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .live-prediction {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
    }
    </style>
"""

class DiseaseDetector:
    def __init__(self):
        self.model = None
        
    @st.cache_resource(show_spinner=False)
    def load_model(self) -> Optional[tf.keras.Model]:
        """Load the disease detection model with error handling."""
        try:
            if not os.path.exists('models'):
                os.makedirs('models')
                
            if not os.path.exists(MODEL_CONFIG['path']):
                with st.spinner('Downloading model... This might take a while...'):
                    gdown.download(MODEL_CONFIG['url'], MODEL_CONFIG['path'], quiet=False)
            
            self.model = tf.keras.models.load_model(MODEL_CONFIG['path'])
            return self.model
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    @staticmethod
    def process_image(image: np.ndarray) -> Optional[np.ndarray]:
        """Process image to model input format."""
        try:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # Handle RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            image = tf.image.resize(image, MODEL_CONFIG['input_size'])
            image = image / 255.0
            return np.expand_dims(image, axis=0)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None
    
    def predict(self, image_array: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Predict disease class for preprocessed image array."""
        prediction = self.model.predict(image_array, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_idx]
        class_name = list(DISEASE_MAPPING.keys())[predicted_class_idx]
        return class_name, confidence, prediction[0]

class VideoProcessor:
    def __init__(self, detector: DiseaseDetector):
        self.detector = detector
        self.prediction_text = ""
        self.confidence = 0.0
        self.frame_skip = 0  # Process every nth frame
        self.current_frame = 0
        
    def process_frame(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process video frames for disease detection."""
        self.current_frame += 1
        if self.current_frame % 30 != 0:  # Process every 30th frame
            return frame
        
        image = frame.to_ndarray(format="bgr24")
        
        # Process the frame
        processed_img = self.detector.process_image(image)
        if processed_img is not None:
            try:
                class_name, confidence, _ = self.detector.predict(processed_img)
                disease_info = DISEASE_MAPPING[class_name]
                
                # Draw prediction on frame
                text = f"{disease_info.display_name}: {confidence:.1%}"
                cv2.putText(
                    image,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if class_name == 'Potato___healthy' else (0, 0, 255),
                    2
                )
                
                # Store prediction for UI
                self.prediction_text = disease_info.display_name
                self.confidence = confidence
                
            except Exception as e:
                print(f"Error in prediction: {str(e)}")
        
        return av.VideoFrame.from_ndarray(image, format="bgr24")

class StreamlitUI:
    def __init__(self):
        self.detector = DiseaseDetector()
        
    def setup_page(self):
        """Configure the Streamlit page."""
        st.set_page_config(
            page_title="Potato Disease Detection",
            page_icon="🥔",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        
    def display_sidebar(self):
        """Display sidebar content."""
        with st.sidebar:
            st.header("About")
            st.markdown("""
            This application uses deep learning to detect diseases in potato leaves. 
            It can identify Early Blight, Late Blight, and Healthy Leaves.
            
            ### Features:
            1. Image Upload Analysis
            2. Live Video Detection
            
            ### Tips for best results:
            - Use clear, well-lit images/video
            - Ensure the leaf is the main focus
            - Hold the camera steady during video
            - Avoid blurry or dark conditions
            """)
            
            with st.expander("Learn About Potato Diseases"):
                for class_name, info in DISEASE_MAPPING.items():
                    st.markdown(f"**{info.display_name}**")
                    st.markdown(info.description)
                    st.markdown("---")
    
    def display_prediction_results(self, class_name: str, confidence: float, 
                                 all_predictions: np.ndarray, processing_time: float):
        """Display the prediction results."""
        disease_info = DISEASE_MAPPING[class_name]
        
        st.subheader("Detection Results")
        
        # Main prediction
        st.markdown(f"### Detected Condition: {disease_info.display_name}")
        st.markdown(f"**Confidence:** {confidence:.1%}")
        
        # Description
        st.markdown(f"**Description:**")
        st.markdown(disease_info.description)
        
        # Detailed analysis
        st.markdown("### Detailed Analysis")
        for idx, (name, info) in enumerate(DISEASE_MAPPING.items()):
            prob = all_predictions[idx]
            st.progress(float(prob))
            st.caption(f"{info.display_name}: {prob:.1%}")
        
        st.caption(f"Processing time: {processing_time:.2f} seconds")
        
        # Recommendations
        if disease_info.recommendations:
            st.markdown("### Recommendations")
            for rec in disease_info.recommendations:
                st.markdown(f"- {rec}")
    
    def run(self):
        """Run the Streamlit application."""
        self.setup_page()
        
        st.title("🥔 Potato Leaf Disease Detection")
        
        # Load model
        with st.spinner("Loading model..."):
            model = self.detector.load_model()
            
        if model is None:
            st.error("Failed to load model. Please refresh the page to try again.")
            return
        
        # Create tabs for different detection methods
        tab1, tab2 = st.tabs(["📷 Image Upload", "🎥 Live Video"])
        
        with tab1:
            st.subheader("Upload a potato leaf image to detect diseases")
            uploaded_file = st.file_uploader(
                "Drag and drop a potato leaf image here",
                type=["jpg", "jpeg", "png"],
                help="Supported formats: JPG, JPEG, PNG"
            )
            
            if uploaded_file:
                try:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        image = Image.open(uploaded_file)
                        st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
                    
                    with col2:
                        start_time = time.time()
                        processed_img = self.detector.process_image(np.array(image))
                        
                        if processed_img is not None:
                            with st.spinner("Analyzing leaf..."):
                                class_name, confidence, all_predictions = self.detector.predict(processed_img)
                            
                            self.display_prediction_results(
                                class_name, confidence, all_predictions,
                                time.time() - start_time
                            )
                            
                except Exception as e:
                    st.error(f"Error analyzing image: {str(e)}")
                    st.info("Please try uploading a different image")
        
        with tab2:
            st.subheader("Live Video Disease Detection")
            st.markdown("""
            Point your camera at a potato leaf to get real-time disease detection.
            Make sure you have good lighting and hold the camera steady.
            """)
            
            # WebRTC configuration
            rtc_configuration = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
            
            # Create video processor
            video_processor = VideoProcessor(self.detector)
            
            # Create WebRTC streamer
            webrtc_ctx = webrtc_streamer(
                key="potato-disease-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_configuration,
                video_processor_factory=lambda: video_processor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            # Display live predictions
            if webrtc_ctx.state.playing:
                st.markdown("### Live Detection Results")
                prediction_placeholder = st.empty()
                while webrtc_ctx.state.playing:
                    if video_processor.prediction_text:
                        prediction_placeholder.markdown(f"""
                        **Detected Condition:** {video_processor.prediction_text}  
                        **Confidence:** {video_processor.confidence:.1%}
                        """)
                    time.sleep(0.1)
        
        self.display_sidebar()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div class='footer'>
                <p>Created with ❤️ by [Your Name] | 
                <a href="https://github.com/yourusername/potato-disease-detection" target="_blank">GitHub</a></p>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    app = StreamlitUI()
    app.run()
