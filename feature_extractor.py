import cv2
import numpy as np
from skimage import feature, measure
from sklearn.preprocessing import StandardScaler
import pandas as pd

class SignatureFeatureExtractor:
    """
    Feature extraction class for signature images.
    Extracts various features including geometric, texture, and statistical features.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def preprocess_image(self, image):
        """
        Preprocess the signature image for feature extraction.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            # Incoming arrays are RGB; convert explicitly to GRAY
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply binary thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Remove small noise components
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Normalize scale and framing: crop to largest contour and resize into fixed canvas
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            pad = max(5, int(0.03 * max(w, h)))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(binary.shape[1], x + w + pad)
            y1 = min(binary.shape[0], y + h + pad)
            cropped = binary[y0:y1, x0:x1]
        else:
            cropped = binary
        
        target_h, target_w = 256, 512
        ch, cw = cropped.shape[:2]
        if ch == 0 or cw == 0:
            normalized = np.zeros((target_h, target_w), dtype=np.uint8)
        else:
            scale = min(target_w / cw, target_h / ch)
            new_w = max(1, int(cw * scale))
            new_h = max(1, int(ch * scale))
            resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            normalized = np.zeros((target_h, target_w), dtype=np.uint8)
            y_off = (target_h - new_h) // 2
            x_off = (target_w - new_w) // 2
            normalized[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        
        return normalized
    
    def extract_geometric_features(self, binary_image):
        """
        Extract geometric features from the signature.
        """
        features = {}
        
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {f'geo_{i}': 0 for i in range(21)}  # Updated for 21 geometric features
        
        # Get the largest contour (main signature)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Basic geometric properties
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Aspect ratio
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # Extent (ratio of contour area to bounding rectangle area)
        extent = float(area) / (w * h) if (w * h) != 0 else 0
        
        # Solidity (ratio of contour area to convex hull area)
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        
        # Compactness
        compactness = (perimeter * perimeter) / area if area != 0 else 0
        
        # Moments
        moments = cv2.moments(main_contour)
        
        # Centroid
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
        else:
            cx, cy = 0, 0
        
        # Hu moments (invariant moments)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Additional geometric features for better forgery detection
        # Convexity defects (indicates stroke smoothness)
        hull = cv2.convexHull(main_contour, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(main_contour, hull)
            if defects is not None and len(defects) > 0:
                # Extract defect depths (4th column)
                defect_depths = defects[:, 0, 3]
                avg_defect_depth = np.mean(defect_depths) / 256.0  # Normalized
                num_defects = len(defects)
            else:
                avg_defect_depth = 0
                num_defects = 0
        else:
            avg_defect_depth = 0
            num_defects = 0
        
        # Curvature (stroke smoothness indicator)
        if len(main_contour) > 2:
            # Approximate contour to reduce noise
            epsilon = 0.02 * cv2.arcLength(main_contour, True)
            approx = cv2.approxPolyDP(main_contour, epsilon, True)
            # Calculate curvature as ratio of approximated to original perimeter
            approx_perimeter = cv2.arcLength(approx, True)
            curvature_ratio = approx_perimeter / perimeter if perimeter > 0 else 0
        else:
            curvature_ratio = 0
        
        # Stroke width variation (forged signatures often have inconsistent stroke width)
        # Use distance transform to estimate stroke width
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
        stroke_width_mean = np.mean(dist_transform[dist_transform > 0]) if np.any(dist_transform > 0) else 0
        stroke_width_std = np.std(dist_transform[dist_transform > 0]) if np.any(dist_transform > 0) else 0
        stroke_width_cv = stroke_width_std / stroke_width_mean if stroke_width_mean > 0 else 0  # Coefficient of variation
        
        # Eccentricity (shape elongation)
        if len(main_contour) >= 5:
            ellipse = cv2.fitEllipse(main_contour)
            a, b = ellipse[1][0] / 2, ellipse[1][1] / 2
            if b > 0:
                eccentricity = np.sqrt(1 - (b * b) / (a * a)) if a >= b else np.sqrt(1 - (a * a) / (b * b))
            else:
                eccentricity = 0
        else:
            eccentricity = 0
        
        features = {
            'geo_0': area,
            'geo_1': perimeter,
            'geo_2': aspect_ratio,
            'geo_3': extent,
            'geo_4': solidity,
            'geo_5': compactness,
            'geo_6': w,  # width
            'geo_7': h,  # height
            'geo_8': cx,  # centroid x
            'geo_9': cy,  # centroid y
            'geo_10': hu_moments[0],
            'geo_11': hu_moments[1],
            'geo_12': hu_moments[2],
            'geo_13': hu_moments[3],
            'geo_14': hu_moments[4],
            'geo_15': float(avg_defect_depth),  # Convexity defect depth
            'geo_16': float(num_defects),  # Number of convexity defects
            'geo_17': float(curvature_ratio),  # Curvature measure
            'geo_18': float(stroke_width_mean),  # Mean stroke width
            'geo_19': float(stroke_width_cv),  # Stroke width variation
            'geo_20': float(eccentricity)  # Shape eccentricity
        }
        
        return features
    
    def extract_texture_features(self, image):
        """
        Extract texture features using Local Binary Patterns and other methods.
        """
        # LBP features
        lbp = feature.local_binary_pattern(image, 24, 8, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize
        
        # GLCM features (simplified)
        glcm = feature.graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                   levels=256, symmetric=True, normed=True)
        
        contrast = feature.graycoprops(glcm, 'contrast').mean()
        dissimilarity = feature.graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
        energy = feature.graycoprops(glcm, 'energy').mean()
        
        features = {}
        
        # Add LBP histogram features
        for i, val in enumerate(lbp_hist[:10]):  # Use first 10 LBP features
            features[f'texture_{i}'] = val
            
        # Add GLCM features
        features['texture_10'] = contrast
        features['texture_11'] = dissimilarity
        features['texture_12'] = homogeneity
        features['texture_13'] = energy
        
        return features
    
    def extract_statistical_features(self, image):
        """
        Extract statistical features from the image.
        """
        features = {}
        
        # Basic statistical measures
        features['stat_0'] = float(np.mean(image))
        features['stat_1'] = float(np.std(image))
        features['stat_2'] = float(np.var(image))
        features['stat_3'] = float(np.median(image))
        features['stat_4'] = float(np.min(image))
        features['stat_5'] = float(np.max(image))
        
        # Skewness and kurtosis approximation
        mean_val = np.mean(image)
        std_val = np.std(image)
        
        if std_val != 0:
            skewness = np.mean(((image - mean_val) / std_val) ** 3)
            kurtosis = np.mean(((image - mean_val) / std_val) ** 4)
        else:
            skewness = 0
            kurtosis = 0
            
        features['stat_6'] = skewness
        features['stat_7'] = kurtosis
        
        # Histogram features
        hist, _ = np.histogram(image, bins=32, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        # Add first 8 histogram bins as features
        for i in range(8):
            features[f'stat_{8+i}'] = hist[i]
            
        return features
    
    def extract_all_features(self, image):
        """
        Extract all features from a signature image.
        """
        # Preprocess the image
        binary_image = self.preprocess_image(image)
        
        # Extract different types of features
        geometric_features = self.extract_geometric_features(binary_image)
        texture_features = self.extract_texture_features(binary_image)
        # Compute statistical features on normalized preprocessed image for consistency
        statistical_features = self.extract_statistical_features(binary_image)
        
        # Combine all features
        all_features = {}
        all_features.update(geometric_features)
        all_features.update(texture_features)
        all_features.update(statistical_features)
        
        return all_features
    
    def extract_features_batch(self, images, labels=None):
        """
        Extract features from a batch of images.
        """
        feature_list = []
        
        for image in images:
            features = self.extract_all_features(image)
            feature_list.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_list)
        
        # Fill any NaN values with 0
        df = df.fillna(0)
        
        # Add labels if provided
        if labels is not None:
            df['label'] = labels
            
        return df
    
    def fit_scaler(self, feature_df):
        """
        Fit the scaler on training features.
        """
        feature_columns = [col for col in feature_df.columns if col != 'label']
        self.scaler.fit(feature_df[feature_columns])
        
    def transform_features(self, feature_df):
        """
        Transform features using the fitted scaler.
        """
        feature_columns = [col for col in feature_df.columns if col != 'label']
        scaled_features = self.scaler.transform(feature_df[feature_columns])
        
        result_df = pd.DataFrame(scaled_features, columns=feature_columns, index=feature_df.index)
        
        if 'label' in feature_df.columns:
            result_df['label'] = feature_df['label']
            
        return result_df
