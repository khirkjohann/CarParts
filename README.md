# Car Parts Detection System

A Streamlit web application that uses deep learning to detect and classify 50 different types of car parts from images.

## Features

- Upload image detection
- Real-time prediction
- 50 different car part classifications
- Confidence score display
- User-friendly interface

## Local Setup

1. Clone the repository
```bash
git clone 
cd 
```

2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up your Google Drive model URL in `.streamlit/secrets.toml`:
```toml
GOOGLE_DRIVE_MODEL_URL = "your-model-url"
```

5. Run the application
```bash
streamlit run app.py
```

## Deployment

This application is deployed on Streamlit Cloud. To deploy your own instance:

1. Fork this repository
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app pointing to your forked repository
4. Add your Google Drive model URL in the Streamlit Cloud secrets management

## Model Information

The model used in this application is trained to detect 50 different car parts:
- AIR COMPRESSOR
- ALTERNATOR
- BATTERY
[... list continues with all 50 parts]

## Requirements

- Python 3.8+
- Streamlit
- TensorFlow
- OpenCV
- Other dependencies listed in requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Model trained using TensorFlow
- Built with Streamlit
- Deployed on Streamlit Cloud
