"""
Model training module for signature verification.
Handles training of individual ML models and preparation of data.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def initialize_models():
    """
    Initialize all ML models with optimized parameters.
    
    Returns:
        dict: Dictionary of model name to model instance
    """
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=100,  # Increased for better accuracy
            max_depth=10,  # Deeper trees for complex patterns
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42, 
            n_jobs=-1  # Use all CPU cores
        ),
        'neural_network': MLPClassifier(
            hidden_layer_sizes=(64, 32),  # Larger network for better learning
            max_iter=500,  # More iterations for convergence
            random_state=42, 
            solver='adam',  # Fast solver
            alpha=0.01,  # L2 regularization
            learning_rate='adaptive',
            early_stopping=True,  # Enable early stopping
            validation_fraction=0.1,
            tol=1e-4,
            warm_start=False
        ),
        'svm': SVC(
            kernel='rbf',  # RBF kernel for non-linear patterns
            probability=True, 
            random_state=42,
            C=1.0,  # Regularization parameter
            gamma='scale',
            max_iter=500  # More iterations
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100,  # More estimators
            learning_rate=0.1,  # Lower learning rate for stability
            max_depth=5,  # Deeper trees
            min_samples_split=5,
            random_state=42
        )
    }
    return models


def train_models(X, y, models, scaler=None, label_encoder=None, progress_callback=None):
    """
    Train all models on the provided dataset.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels array
        models (dict): Dictionary of models to train
        scaler (StandardScaler): Optional pre-fitted scaler
        label_encoder (LabelEncoder): Optional pre-fitted label encoder
        progress_callback (callable): Optional callback for progress updates
        
    Returns:
        tuple: (trained_models, scaler, label_encoder)
    """
    # Initialize scaler and label encoder if not provided
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X)
    
    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(y)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Encode labels
    y_encoded = label_encoder.transform(y)
    
    # Train all models with progress tracking
    trained_models = {}
    model_names_display = {
        'random_forest': 'Random Forest',
        'neural_network': 'Neural Network',
        'svm': 'Support Vector Machine',
        'gradient_boosting': 'Gradient Boosting'
    }
    
    for idx, (model_name, model) in enumerate(models.items(), 1):
        if progress_callback:
            progress_callback(f"Training {model_names_display.get(model_name, model_name)} ({idx}/{len(models)})...")
        
        model.fit(X_scaled, y_encoded)
        trained_models[model_name] = model
    
    return trained_models, scaler, label_encoder


def prepare_and_train(reference_image, feature_extractor):
    """
    Complete pipeline: prepare dataset and train all models.
    
    Args:
        reference_image (np.ndarray): Reference signature image
        feature_extractor: Feature extractor instance
        
    Returns:
        tuple: (models, scaler, label_encoder, reference_features)
    """
    from dataset import prepare_training_dataset
    
    # Prepare training dataset
    X, y = prepare_training_dataset(reference_image, feature_extractor)
    
    # Initialize models
    models = initialize_models()
    
    # Train models
    trained_models, scaler, label_encoder = train_models(X, y, models)
    
    # Extract reference features
    reference_features = feature_extractor.extract_all_features(reference_image)
    
    return trained_models, scaler, label_encoder, reference_features
