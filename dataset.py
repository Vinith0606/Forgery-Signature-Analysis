"""
Dataset generation and augmentation for signature verification.
Handles creation of training variations from reference signatures.
"""

import cv2
import numpy as np


def generate_reference_variations(image, num_variations=15):
    """
    Generate consistent variations of the reference signature.
    Creates genuine-like variations with minimal natural variations.
    These should be VERY similar to the original (same person signing).
    
    Args:
        image (np.ndarray): Original signature image
        num_variations (int): Number of variations to generate
        
    Returns:
        list: List of variation images
    """
    variations = []
    
    # Set seed for consistent results
    np.random.seed(42)
    
    h, w = image.shape[:2]
    
    for i in range(num_variations):
        variation = image.copy()
        np.random.seed(42 + i)
        
        # Apply only ONE small transformation per variation to keep them very similar
        transform_type = i % 4
        
        if transform_type == 0:
            # Very small rotation (-1 to +1 degrees) - natural hand variation
            angle = np.random.uniform(-1, 1)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            variation = cv2.warpAffine(variation, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        elif transform_type == 1:
            # Tiny translation (-1 to +1 pixels) - natural positioning
            tx = np.random.uniform(-1, 1)
            ty = np.random.uniform(-1, 1)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            variation = cv2.warpAffine(variation, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        elif transform_type == 2:
            # Minimal brightness variation (-3 to +3) - natural lighting
            brightness = np.random.uniform(-3, 3)
            variation = cv2.convertScaleAbs(variation, alpha=1, beta=brightness)
        else:
            # Very minimal noise (natural pen variation)
            noise = np.random.normal(0, 0.5, variation.shape).astype(np.float32)
            variation = np.clip(variation.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        variations.append(variation)
    
    return variations


def generate_forged_variations(image, num_variations=20):
    """
    Generate consistent forged variations.
    Creates forged-like variations with SIGNIFICANT differences from original.
    These should be clearly different (different person trying to copy).
    
    Args:
        image (np.ndarray): Original signature image
        num_variations (int): Number of variations to generate
        
    Returns:
        list: List of forged variation images
    """
    variations = []
    
    # Set seed for consistent results
    np.random.seed(100)  # Different base seed from genuine variations
    
    h, w = image.shape[:2]
    
    for i in range(num_variations):
        variation = image.copy()
        np.random.seed(100 + i)
        
        # Apply MULTIPLE significant transformations to make clear forgeries
        # Type 1: Geometric distortions (30% of forgeries)
        if i < num_variations * 0.3:
            # Significant rotation (-15 to +15 degrees) - wrong angle
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            variation = cv2.warpAffine(variation, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            # Scale variation (0.75 to 1.25) - wrong size
            scale = np.random.uniform(0.75, 1.25)
            M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
            variation = cv2.warpAffine(variation, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            # Translation (wrong position)
            tx = np.random.uniform(-w*0.1, w*0.1)
            ty = np.random.uniform(-h*0.1, h*0.1)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            variation = cv2.warpAffine(variation, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Type 2: Perspective/affine distortion (30% of forgeries)
        elif i < num_variations * 0.6:
            # Affine transformation with safe point generation
            max_offset = min(w, h) * 0.15  # Larger offset (15%) for more distortion
            
            # Source points (three corners of a triangle)
            pts1 = np.float32([
                [w * 0.1, h * 0.1],      # Top-left area
                [w * 0.9, h * 0.1],      # Top-right area
                [w * 0.5, h * 0.9]       # Bottom-center area
            ])
            
            # Destination points with larger random offset
            offset_x = np.random.uniform(-max_offset, max_offset, 3)
            offset_y = np.random.uniform(-max_offset, max_offset, 3)
            
            pts2 = pts1.copy()
            pts2[:, 0] += offset_x
            pts2[:, 1] += offset_y
            
            # Ensure points stay within image bounds
            pts2[:, 0] = np.clip(pts2[:, 0], 0, w - 1)
            pts2[:, 1] = np.clip(pts2[:, 1], 0, h - 1)
            
            # Ensure points are still in float32 format and correct shape
            pts1 = pts1.reshape(3, 2).astype(np.float32)
            pts2 = pts2.reshape(3, 2).astype(np.float32)
            
            try:
                M = cv2.getAffineTransform(pts1, pts2)
                variation = cv2.warpAffine(variation, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            except cv2.error:
                # Fallback: use simpler transformation if affine fails
                angle = np.random.uniform(-10, 10)
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, np.random.uniform(0.8, 1.2))
                variation = cv2.warpAffine(variation, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Type 3: Style differences (40% of forgeries)
        else:
            # Heavy blur (different pen pressure/style)
            blur_size = np.random.choice([7, 9, 11])
            variation = cv2.GaussianBlur(variation, (blur_size, blur_size), 0)
            
            # Significant noise (shaky hand, different technique)
            noise = np.random.normal(0, 8, variation.shape).astype(np.float32)
            variation = np.clip(variation.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Common distortions for all types - make them clearly different
        # Brightness/contrast changes (wrong lighting/pen)
        brightness = np.random.uniform(-30, 30)
        contrast = np.random.uniform(0.6, 1.4)
        variation = cv2.convertScaleAbs(variation, alpha=contrast, beta=brightness)
        
        # Additional noise (different writing style)
        noise = np.random.normal(0, 5, variation.shape).astype(np.float32)
        variation = np.clip(variation.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        variations.append(variation)
    
    return variations


def prepare_training_dataset(reference_image, feature_extractor):
    """
    Prepare complete training dataset from a single reference signature.
    
    Args:
        reference_image (np.ndarray): Original reference signature
        feature_extractor: Feature extractor instance
        
    Returns:
        tuple: (X, y) - Features array and labels array
    """
    # Create genuine variations (increased for better training)
    genuine_variations = generate_reference_variations(reference_image, num_variations=15)
    
    # Create forged variations (increased for better training)
    forged_variations = generate_forged_variations(reference_image, num_variations=20)
    
    # Prepare training dataset
    training_features = []
    training_labels = []
    
    # Add the original signature as genuine
    ref_features = feature_extractor.extract_all_features(reference_image)
    training_features.append(list(ref_features.values()))
    training_labels.append('genuine')
    
    # Add genuine variations
    for variation in genuine_variations:
        features = feature_extractor.extract_all_features(variation)
        training_features.append(list(features.values()))
        training_labels.append('genuine')
    
    # Add forged variations
    for variation in forged_variations:
        features = feature_extractor.extract_all_features(variation)
        training_features.append(list(features.values()))
        training_labels.append('forged')
    
    # Convert to numpy arrays
    X = np.array(training_features)
    y = np.array(training_labels)
    
    return X, y
