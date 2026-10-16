"""
Utility functions for signature verification system.
Includes image loading, display, and helper functions.
"""

import numpy as np
from PIL import Image
import streamlit as st


def load_image(uploaded_file):
    """
    Load and convert uploaded image to numpy array.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        np.ndarray: Image array or None if error
    """
    try:
        image = Image.open(uploaded_file)
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Convert to numpy array
        image_array = np.array(image)
        return image_array
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None


def display_image_with_info(image, title, col):
    """
    Display image with information in a column.
    
    Args:
        image (np.ndarray): Image to display
        title (str): Title/caption for the image
        col: Streamlit column object
    """
    with col:
        st.image(image, caption=title, use_column_width=True)
        st.write(f"**Shape:** {image.shape}")
        st.write(f"**Size:** {image.shape[0]} x {image.shape[1]} pixels")


def create_image_hash(image):
    """
    Create a hash of an image for caching purposes.
    
    Args:
        image (np.ndarray): Image array
        
    Returns:
        int: Hash value
    """
    return hash(image.tobytes())


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 2rem;
            color: #1f77b4;
        }
        .section-header {
            font-size: 1.5rem;
            font-weight: bold;
            margin-top: 2rem;
            margin-bottom: 1rem;
            color: #2e8b57;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .prediction-genuine {
            background-color: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #c3e6cb;
        }
        .prediction-forged {
            background-color: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #f5c6cb;
            text-align: center;
        }
        .reference-info {
            background-color: #e7f3ff;
            color: #0c5460;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #b8daff;
        }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables."""
    from feature_extractor import SignatureFeatureExtractor
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from train import initialize_models
    
    if 'feature_extractor' not in st.session_state:
        st.session_state.feature_extractor = SignatureFeatureExtractor()
    
    if 'reference_features' not in st.session_state:
        st.session_state.reference_features = None
    
    if 'reference_image' not in st.session_state:
        st.session_state.reference_image = None
    
    if 'reference_set' not in st.session_state:
        st.session_state.reference_set = False
    
    if 'scaler' not in st.session_state:
        st.session_state.scaler = StandardScaler()
    
    if 'ml_models' not in st.session_state:
        st.session_state.ml_models = initialize_models()
    
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    
    if 'label_encoder' not in st.session_state:
        st.session_state.label_encoder = LabelEncoder()
    
    if 'last_reference_hash' not in st.session_state:
        st.session_state.last_reference_hash = None
    
    if 'verification_cache' not in st.session_state:
        st.session_state.verification_cache = {}
    
    if 'ensemble' not in st.session_state:
        st.session_state.ensemble = None
