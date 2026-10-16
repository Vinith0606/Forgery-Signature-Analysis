"""
Database operations for signature storage and retrieval.
Handles SQLite database interactions for signature management.
"""

import sqlite3
import base64
import cv2
import numpy as np
import streamlit as st


def init_database():
    """Initialize SQLite database for storing signatures."""
    conn = sqlite3.connect('signatures.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signatures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        )
    ''')
    
    conn.commit()
    conn.close()


def save_signature_to_db(name, image_array, description=""):
    """
    Save signature to database.
    
    Args:
        name (str): Name of the signature owner
        image_array (np.ndarray): Image array to save (should be RGB format)
        description (str): Optional description
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure image is in correct format for saving
        # Images from PIL/Streamlit are RGB, but cv2.imencode expects BGR
        image_to_save = image_array.copy()
        
        # Check if image is RGB (3 channels) and convert to BGR for cv2.imencode
        if len(image_to_save.shape) == 3 and image_to_save.shape[2] == 3:
            # Convert RGB to BGR for OpenCV encoding
            image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
        
        # Convert image to base64 string using PNG format (lossless)
        _, buffer = cv2.imencode('.png', image_to_save)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        conn = sqlite3.connect('signatures.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signatures (name, image_data, description)
            VALUES (?, ?, ?)
        ''', (name, image_base64, description))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving signature: {e}")
        return False


def load_signatures_from_db():
    """
    Load all signatures from database.
    
    Returns:
        list: List of tuples containing signature data
    """
    try:
        conn = sqlite3.connect('signatures.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, image_data, created_at, description
            FROM signatures
            ORDER BY created_at DESC
        ''')
        
        signatures = cursor.fetchall()
        conn.close()
        return signatures
    except Exception as e:
        st.error(f"Error loading signatures: {e}")
        return []


def delete_signature_from_db(signature_id):
    """
    Delete signature from database.
    
    Args:
        signature_id (int): ID of the signature to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect('signatures.db')
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM signatures WHERE id = ?', (signature_id,))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error deleting signature: {e}")
        return False


def base64_to_image(base64_string):
    """
    Convert base64 string back to image array.
    
    Args:
        base64_string (str): Base64 encoded image string
        
    Returns:
        np.ndarray: Image array or None if error
    """
    try:
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Normalize to RGB to match other image loading paths
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) if image_bgr is not None else None
        return image_rgb
    except Exception as e:
        st.error(f"Error converting image: {e}")
        return None
