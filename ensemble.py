"""
Ensemble model for signature verification.
Combines predictions from multiple ML algorithms using weighted voting.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class SignatureEnsemble:
    """
    Ensemble classifier for signature verification.
    Combines multiple ML models using weighted voting.
    """
    
    def __init__(self, models, scaler, label_encoder, weights=None):
        """
        Initialize ensemble with trained models.
        
        Args:
            models (dict): Dictionary of trained models
            scaler: Fitted StandardScaler
            label_encoder: Fitted LabelEncoder
            weights (dict): Optional custom weights for each model
        """
        self.models = models
        self.scaler = scaler
        self.label_encoder = label_encoder
        
        # Default weights
        self.weights = weights or {
            'Random Forest': 0.30,
            'Neural Network': 0.25,
            'Support Vector Machine': 0.25,
            'Gradient Boosting': 0.20
        }
        
        # Model name mapping
        self.model_names = {
            'random_forest': 'Random Forest',
            'neural_network': 'Neural Network', 
            'svm': 'Support Vector Machine',
            'gradient_boosting': 'Gradient Boosting'
        }
    
    def predict_single(self, test_features, reference_features=None):
        """
        Predict whether a signature is genuine or forged.
        Uses similarity-based approach with ML models as secondary validation.
        
        Args:
            test_features (dict): Features extracted from test signature
            reference_features (dict): Optional reference features for similarity
            
        Returns:
            dict: Prediction results with confidence and details
        """
        # Convert features to array
        test_features_array = np.array([list(test_features.values())])
        
        # Scale features
        test_features_scaled = self.scaler.transform(test_features_array)
        
        # PRIMARY DECISION: Similarity-based comparison (most reliable)
        similarity_metrics = {}
        similarity_score = 0.0
        is_genuine_by_similarity = False
        
        if reference_features is not None:
            ref_features_array = np.array([list(reference_features.values())])
            ref_features_scaled = self.scaler.transform(ref_features_array)
            
            # Calculate multiple similarity metrics
            cosine_sim = cosine_similarity(test_features_scaled, ref_features_scaled)[0][0]
            euclidean_dist = euclidean_distances(test_features_scaled, ref_features_scaled)[0][0]
            
            # Normalize euclidean distance to 0-1 scale (inverse similarity)
            # Assuming max distance is around 20 for scaled features
            max_expected_dist = 20.0
            normalized_euclidean = 1.0 - min(euclidean_dist / max_expected_dist, 1.0)
            
            # Calculate feature difference percentage
            feature_diff = np.abs(test_features_scaled - ref_features_scaled).mean()
            feature_similarity = 1.0 - min(feature_diff / 2.0, 1.0)  # Normalize
            
            # Combined similarity score (weighted average)
            similarity_score = (cosine_sim * 0.5 + normalized_euclidean * 0.3 + feature_similarity * 0.2)
            
            similarity_metrics = {
                'cosine_similarity': float(cosine_sim),
                'euclidean_distance': float(euclidean_dist),
                'feature_similarity': float(feature_similarity),
                'combined_similarity': float(similarity_score)
            }
            
            # Similarity-based decision: High similarity (>0.85) = Genuine, Low (<0.65) = Forged
            if similarity_score > 0.85:
                is_genuine_by_similarity = True
            elif similarity_score < 0.65:
                is_genuine_by_similarity = False
            else:
                # Gray zone - use ML models to decide
                is_genuine_by_similarity = None
        
        # SECONDARY DECISION: ML model predictions (for validation and gray zone)
        algorithms_results = {}
        
        for model_key, model_name in self.model_names.items():
            model = self.models[model_key]
            
            # Get prediction
            prediction = model.predict(test_features_scaled)[0]
            
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(test_features_scaled)[0]
                confidence = np.max(probabilities) * 100
                
                # Get class probabilities
                classes = self.label_encoder.classes_
                prob_dict = dict(zip(classes, probabilities))
            else:
                # For models without predict_proba, use decision function or default
                confidence = 75.0  # Default confidence
                prob_dict = {'genuine': 0.5, 'forged': 0.5}
            
            # Convert prediction back to label
            prediction_label = self.label_encoder.inverse_transform([prediction])[0]
            
            algorithms_results[model_name] = {
                'prediction': prediction_label.title(),
                'confidence': confidence,
                'probabilities': prob_dict
            }
        
        # Calculate ML ensemble votes
        genuine_votes = 0
        total_confidence = 0
        genuine_confidences = []
        forged_confidences = []
        
        for algo, result in algorithms_results.items():
            weight = self.weights[algo]
            if result['prediction'] == 'Genuine':
                genuine_votes += weight
                genuine_confidences.append(result['confidence'] * weight)
            else:
                forged_confidences.append(result['confidence'] * weight)
            total_confidence += result['confidence'] * weight
        
        avg_genuine_conf = np.mean(genuine_confidences) if genuine_confidences else 0
        avg_forged_conf = np.mean(forged_confidences) if forged_confidences else 0
        
        # FINAL DECISION: Combine similarity and ML predictions
        if reference_features is not None:
            # Use similarity as primary decision maker
            if is_genuine_by_similarity is not None:
                # Clear similarity decision
                is_genuine = is_genuine_by_similarity
                # Adjust confidence based on similarity strength
                if is_genuine:
                    # High similarity = high confidence in genuine
                    ensemble_confidence = min(95.0, similarity_score * 100)
                else:
                    # Low similarity = high confidence in forged
                    ensemble_confidence = min(95.0, (1.0 - similarity_score) * 100)
            else:
                # Gray zone: Use ML models with similarity adjustment
                # If ML says genuine AND similarity is moderate, trust ML
                # If ML says forged OR similarity is low, mark as forged
                ml_says_genuine = genuine_votes > 0.5
                
                if ml_says_genuine and similarity_score > 0.70:
                    is_genuine = True
                    ensemble_confidence = (similarity_score * 0.6 + (genuine_votes * 100) * 0.4)
                else:
                    is_genuine = False
                    ensemble_confidence = ((1.0 - similarity_score) * 0.6 + (1.0 - genuine_votes) * 100 * 0.4)
        else:
            # No reference: Use only ML models (fallback)
            is_genuine = genuine_votes > 0.5
            ensemble_confidence = total_confidence
        
        ensemble_prediction = 'Genuine' if is_genuine else 'Forged'
        
        # Compile results
        results = {
            'ensemble_prediction': ensemble_prediction,
            'ensemble_confidence': ensemble_confidence,
            'is_genuine': ensemble_prediction == 'Genuine',
            'algorithms': algorithms_results,
            'similarity_metrics': similarity_metrics,
            'voting_details': {
                'genuine_votes': genuine_votes,
                'total_votes': 1.0,
                'weights': self.weights
            }
        }
        
        return results
    
    def set_weights(self, weights):
        """
        Update ensemble weights.
        
        Args:
            weights (dict): New weights for each algorithm
        """
        self.weights = weights
