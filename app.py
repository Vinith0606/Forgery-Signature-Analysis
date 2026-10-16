import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import io

# Import custom modules
from database import (
    init_database, 
    save_signature_to_db, 
    load_signatures_from_db, 
    delete_signature_from_db, 
    base64_to_image
)
from utils import (
    load_image, 
    apply_custom_css, 
    initialize_session_state,
    create_image_hash
)
from train import prepare_and_train
from ensemble import SignatureEnsemble
from visualization import (
    create_performance_analytics,
    create_accuracy_comparison_chart,
    create_precision_recall_f1_chart,
    create_confusion_matrix_heatmap,
    create_roc_curves_chart,
    create_robustness_analysis_chart,
    create_confidence_visualization
)

# Configure page
st.set_page_config(
    page_title="Digital Signature Verification System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# Initialize session state
initialize_session_state()

# Initialize database
init_database()

def set_reference_signature(signature_image):
    """Process ONE original signature and automatically prepare ML models."""
    try:
        # Create a hash of the image to check if it's the same as before
        image_hash = create_image_hash(signature_image)
        
        # If same image, skip processing
        if st.session_state.last_reference_hash == image_hash and st.session_state.models_trained:
            st.success("✅ Using cached reference signature!")
            return True
        
        # Create a progress container
        progress_placeholder = st.empty()
        
        with progress_placeholder.container():
            st.info("🔄 Step 1/3: Processing signature image...")
        
        # Store the original signature
        st.session_state.reference_image = signature_image.copy()
        st.session_state.last_reference_hash = image_hash
        
        # Automatically generate training data and train models
        with progress_placeholder.container():
            st.info("🔄 Step 2/3: Generating training variations...")
        
        trained_models, scaler, label_encoder, reference_features = prepare_and_train(
            signature_image, 
            st.session_state.feature_extractor
        )
        
        with progress_placeholder.container():
            st.info("🔄 Step 3/3: Finalizing ensemble model...")
        
        # Update session state
        st.session_state.ml_models = trained_models
        st.session_state.scaler = scaler
        st.session_state.label_encoder = label_encoder
        st.session_state.reference_features = reference_features
        
        # Create ensemble
        st.session_state.ensemble = SignatureEnsemble(
            trained_models, 
            scaler, 
            label_encoder
        )
        
        st.session_state.reference_set = True
        st.session_state.models_trained = True
        
        # Clear progress and show success
        progress_placeholder.empty()
        st.success("✅ Reference signature processed! Ready for verification.")
        return True
        
    except Exception as e:
        st.error(f"❌ Error processing signature: {e}")
        return False

def compare_signature(test_signature):
    """Compare a test signature using trained ensemble model."""
    if not st.session_state.models_trained or st.session_state.ensemble is None:
        st.error("Models not trained. Please set a reference signature first.")
        return None
    
    try:
        # Create hash of test signature for caching
        test_hash = create_image_hash(test_signature)
        
        # Check if we already have results for this signature
        cache_key = f"{st.session_state.last_reference_hash}_{test_hash}"
        if cache_key in st.session_state.verification_cache:
            return st.session_state.verification_cache[cache_key]
        
        # Extract features from test signature
        test_features = st.session_state.feature_extractor.extract_all_features(test_signature)
        
        # Get prediction from ensemble
        results = st.session_state.ensemble.predict_single(
            test_features, 
            st.session_state.reference_features
        )
        
        # Cache the results
        st.session_state.verification_cache[cache_key] = results
        
        return results
        
    except Exception as e:
        st.error(f"Error comparing signatures: {e}")
        return None

# Main application
def main():
    st.markdown('<div class="main-header">Digital Signature Verification System</div>', unsafe_allow_html=True)
    st.markdown("**Reference-Based Signature Authentication Using Multiple ML Algorithms**")
    
    # Sidebar for navigation and control center
    st.sidebar.title("Control Center")
    
    # Signature Management Section
    st.sidebar.markdown("###  Signature Database")
    
    with st.sidebar.expander("➕ Add New Signature"):
        with st.form("add_signature_form"):
            signature_name = st.text_input("Signature Name*", placeholder="Enter person's name")
            signature_description = st.text_area("Description (Optional)", placeholder="Additional notes")
            signature_file = st.file_uploader("Upload Signature", type=['png', 'jpg', 'jpeg', 'bmp'])
            
            if st.form_submit_button("💾 Save Signature"):
                if signature_name and signature_file:
                    image = load_image(signature_file)
                    if image is not None:
                        if save_signature_to_db(signature_name, image, signature_description):
                            st.success(f" Signature for '{signature_name}' saved!")
                            st.rerun()
                        else:
                            st.error(" Failed to save signature")
                    else:
                        st.error(" Invalid image file")
                else:
                    st.error(" Please provide name and signature file")
    
    # View Stored Signatures
    with st.sidebar.expander(" Manage Stored Signatures"):
        signatures = load_signatures_from_db()
        
        if signatures:
            st.write(f"**Total Signatures:** {len(signatures)}")
            
            for sig_id, name, _, created_at, description in signatures:
                col1, col2 = st.columns([2.5, 1.5])
                
                with col1:
                    st.write(f"**{name}**")
                    if description:
                        st.caption(description)
                    st.caption(f"Added: {created_at[:10]}")
                
                with col2:
                    if st.button("Delete", key=f"delete_{sig_id}", help="Delete signature", type="secondary", use_container_width=True):
                        if delete_signature_from_db(sig_id):
                            st.success("Deleted!")
                            st.rerun()
                
                st.markdown("---")
        else:
            st.info("No signatures stored yet")
    
    # How it works section
    st.sidebar.markdown("### ℹ How It Works")
    st.sidebar.info("""
    **Manual Verification:**
    1️⃣ Upload original signature
    2️⃣ Upload signature to verify
    3️⃣ Get ML analysis results
    
    **Stored Verification:**
    1️⃣ Select stored signature
    2️⃣ Upload signature to verify
    3️⃣ Get ML analysis results
    
    **Live Verification:**
    1️⃣ Select stored signature
    2️⃣ Capture with camera
    3️⃣ Get ML analysis results
    """)
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["Stored Verification", "Manual Verification", "Live Verification", "System Info"])
    
    with tab1:
        stored_verification_tab()
    
    with tab2:
        verification_tab()
    
    with tab3:
        live_verification_tab()
    
    with tab4:
        system_info_tab()


def live_verification_tab():
    st.markdown('<div class="section-header">Live Signature Verification</div>', unsafe_allow_html=True)
    
    # Load stored signatures
    signatures = load_signatures_from_db()
    
    if not signatures:
        st.warning(" No signatures stored yet. Please add signatures using the Control Center in the sidebar.")
        st.info(" Use the ' Add New Signature' section in the sidebar to store signatures first.")
        return
    
    # Create two main columns
    col1, col2 = st.columns([1, 1])
    
    # Left Column - Select Stored Signature for Reference
    with col1:
        left_spacer, center_content, right_spacer = st.columns([0.1, 0.8, 0.1])
        
        with center_content:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.subheader(" Step 1: Select Reference Signature")
            st.write("Choose a signature from database")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Create signature selection dropdown with None option
            signature_options = {"-- Select a signature --": None}
            signature_options.update({f"{name} (ID: {sig_id})": (sig_id, name, image_data, description) 
                                    for sig_id, name, image_data, created_at, description in signatures})
            
            selected_signature = st.selectbox(
                "Choose reference signature:",
                options=list(signature_options.keys()),
                index=0,
                key="live_signature_selector"
            )
            
            if selected_signature and signature_options[selected_signature] is not None:
                sig_id, name, image_data, description = signature_options[selected_signature]
                
                # Display selected signature
                stored_image = base64_to_image(image_data)
                if stored_image is not None:
                    st.image(stored_image, caption=f"Reference: {name}", use_column_width=True)
                    
                    # Center the image info
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    st.write(f"**Name:** {name}")
                    if description:
                        st.write(f"**Description:** {description}")
                    st.write(f"**Shape:** {stored_image.shape}")
                    st.write(f"**Size:** {stored_image.shape[0]} x {stored_image.shape[1]} pixels")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Process button for stored signature
                    if st.button(" Process Reference Signature", type="secondary", use_container_width=True, key="process_live_reference"):
                        with st.spinner("Processing reference signature..."):
                            success = set_reference_signature(stored_image)
                        if success:
                            st.success(" Reference signature processed!")
                        else:
                            st.error(" Failed to process signature")
    
    # Right Column - Live Camera Input
    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.subheader(" Step 2: Live Camera Verification")
        st.write("Capture signature using camera")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Initialize camera state
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        if 'captured_image' not in st.session_state:
            st.session_state.captured_image = None
        if 'capture_requested' not in st.session_state:
            st.session_state.capture_requested = False
        if 'camera_type' not in st.session_state:
            st.session_state.camera_type = 'streamlit'
        
        # Only use system camera (Streamlit camera)
        
        # Camera control buttons (stacked vertically to avoid nesting)
        if not st.session_state.camera_active:
            if st.button(" Start Camera", type="primary", use_container_width=True, key="start_camera"):
                st.session_state.camera_active = True
                st.rerun()
        else:
            if st.button(" Stop Camera", type="secondary", use_container_width=True, key="stop_camera"):
                st.session_state.camera_active = False
                st.rerun()
            
            if st.button(" Capture Image", type="primary", use_container_width=True, key="capture_btn"):
                st.session_state.capture_requested = True
        
        # System Camera Input (Streamlit Camera Only)
        camera_image = None
        
        if st.session_state.camera_active:
            st.info("🎥 System Camera Active")
            camera_image = st.camera_input("Camera View", key="live_camera", disabled=False)
            
            # Auto-capture when capture button is pressed
            if hasattr(st.session_state, 'capture_requested') and st.session_state.capture_requested and camera_image is not None:
                st.session_state.captured_image = camera_image
                st.session_state.capture_requested = False
                st.success(" Image captured successfully!")
            
            # Only clear results if explicitly requested
            if 'clear_live_results' in st.session_state and st.session_state.clear_live_results:
                if hasattr(st.session_state, 'verification_results'):
                    del st.session_state.verification_results
                st.session_state.clear_live_results = False
            
            # Display captured image and verification options
            display_image = st.session_state.captured_image if st.session_state.captured_image is not None else camera_image
            
            if display_image is not None:
                # Convert camera image to numpy array
                # Streamlit's camera_input returns an UploadedFile; use its bytes
                try:
                    file_bytes = display_image.getvalue()
                except AttributeError:
                    # Fallback if already bytes-like
                    file_bytes = display_image
                camera_pil = Image.open(io.BytesIO(file_bytes))
                if camera_pil.mode != 'RGB':
                    camera_pil = camera_pil.convert('RGB')
                live_image = np.array(camera_pil)
                
                st.image(live_image, caption="Live Captured Signature", use_column_width=True)
                
                # Center the image info
                st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                st.write(f"**Shape:** {live_image.shape}")
                st.write(f"**Size:** {live_image.shape[0]} x {live_image.shape[1]} pixels")
                st.markdown("</div>", unsafe_allow_html=True)
                
                if st.session_state.models_trained:
                    # Live detection toggle
                    live_detection = st.checkbox(" Enable Live Detection", key="live_detection_toggle")
                    
                    if live_detection and st.session_state.camera_active and display_image is not None:
                        # Continuous verification for live detection
                        with st.spinner("Live analyzing..."):
                            results = compare_signature(live_image)
                            if results:
                                st.session_state.verification_results = results
                                
                                # Show quick result indicator
                                if results['is_genuine']:
                                    st.success(f" GENUINE - {results['ensemble_confidence']:.1f}% confidence")
                                else:
                                    st.error(f" FORGED - {results['ensemble_confidence']:.1f}% confidence")
                    
                    # Manual verification button
                    if st.button(" Verify Signature", type="primary", use_container_width=True, key="verify_live"):
                        with st.spinner("Analyzing signature..."):
                            results = compare_signature(live_image)
                            if results:
                                st.session_state.verification_results = results
                                st.success("Analysis complete!")
                    
                    if st.button(" Clear Results", type="secondary", use_container_width=True, key="clear_live"):
                        st.session_state.clear_live_results = True
                        st.rerun()
                else:
                    st.warning(" Please select and process a reference signature first")
    
    # Results Section (same as other verification methods)
    display_verification_results()

def stored_verification_tab():
    st.markdown('<div class="section-header">Stored Signature Verification</div>', unsafe_allow_html=True)
    
    # Load stored signatures
    signatures = load_signatures_from_db()
    
    if not signatures:
        st.warning(" No signatures stored yet. Please add signatures using the Control Center in the sidebar.")
        st.info(" Use the ' Add New Signature' section in the sidebar to store signatures first.")
        return
    
    # Create two main columns
    col1, col2 = st.columns([1, 1])
    
    # Left Column - Select Stored Signature
    with col1:
        left_spacer, center_content, right_spacer = st.columns([0.1, 0.8, 0.1])
        
        with center_content:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.subheader(" Step 1: Select Stored Signature")
            st.write("Choose a signature from database")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Create signature selection dropdown with None option
            signature_options = {"-- Select a signature --": None}
            signature_options.update({f"{name} (ID: {sig_id})": (sig_id, name, image_data, description) 
                                    for sig_id, name, image_data, created_at, description in signatures})
            
            selected_signature = st.selectbox(
                "Choose stored signature:",
                options=list(signature_options.keys()),
                index=0
            )
            
            if selected_signature and signature_options[selected_signature] is not None:
                sig_id, name, image_data, description = signature_options[selected_signature]
                
                # Display selected signature
                stored_image = base64_to_image(image_data)
                if stored_image is not None:
                    st.image(stored_image, caption=f"Stored Signature: {name}", use_column_width=True)
                    
                    # Center the image info
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    st.write(f"**Name:** {name}")
                    if description:
                        st.write(f"**Description:** {description}")
                    st.write(f"**Shape:** {stored_image.shape}")
                    st.write(f"**Size:** {stored_image.shape[0]} x {stored_image.shape[1]} pixels")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Process button for stored signature
                    if st.button(" Process Stored Signature", type="secondary", use_container_width=True, key="process_stored"):
                        with st.spinner("Processing stored signature..."):
                            success = set_reference_signature(stored_image)
                        if success:
                            st.success(" Stored signature processed!")
                        else:
                            st.error(" Failed to process signature")
    
    # Right Column - Verification Signature Section
    with col2:
        left_spacer2, center_content2, right_spacer2 = st.columns([0.1, 0.8, 0.1])
        
        with center_content2:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.subheader(" Step 2: Verify Signature")
            st.write("Upload signature to verify against stored one")
            st.markdown("</div>", unsafe_allow_html=True)
            
            test_file = st.file_uploader(
                "Choose signature to verify",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                key="stored_verification_uploader"
            )
            
            # Clear results if verification image is removed
            if test_file is None and hasattr(st.session_state, 'verification_results'):
                del st.session_state.verification_results
            
            if test_file is not None:
                test_image = load_image(test_file)
                if test_image is not None:
                    st.image(test_image, caption="Signature to Verify", use_column_width=True)
                    
                    # Center the image info
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    st.write(f"**Shape:** {test_image.shape}")
                    st.write(f"**Size:** {test_image.shape[0]} x {test_image.shape[1]} pixels")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    if st.session_state.models_trained:
                        if st.button(" Verify Against Stored Signature", type="primary", use_container_width=True):
                            with st.spinner("Analyzing signature..."):
                                results = compare_signature(test_image)
                                if results:
                                    st.session_state.verification_results = results
                                    st.success("✅ Verification complete!")
                    else:
                        st.warning(" Please select and process a stored signature first")
    
    # Results Section (same as manual verification)
    display_verification_results()

def verification_tab():
    st.markdown('<div class="section-header">Signature Verification System</div>', unsafe_allow_html=True)
    
    # Create two main columns for original and verification signatures
    col1, col2 = st.columns([1, 1])
    
    # Left Column - Original Signature Section
    with col1:
        # Center the content using columns
        left_spacer, center_content, right_spacer = st.columns([0.1, 0.8, 0.1])
        
        with center_content:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.subheader(" Step 1: Original Signature")
            st.write("Upload ONE genuine signature")
            st.markdown("</div>", unsafe_allow_html=True)
            
            reference_file = st.file_uploader(
                "Choose original signature image",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                key="reference_uploader_main"
            )
            
            # Clear results if reference image is removed
            if reference_file is None and hasattr(st.session_state, 'verification_results'):
                del st.session_state.verification_results
                st.session_state.reference_set = False
                st.session_state.models_trained = False
            
            if reference_file is not None:
                reference_image = load_image(reference_file)
                if reference_image is not None:
                    st.image(reference_image, caption="Original Signature", use_column_width=True)
                    
                    # Center the image info
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    st.write(f"**Shape:** {reference_image.shape}")
                    st.write(f"**Size:** {reference_image.shape[0]} x {reference_image.shape[1]} pixels")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Process button for reference signature
                    if st.button(" Process Reference Signature", type="secondary", use_container_width=True, key="process_reference"):
                        with st.spinner("Processing reference signature..."):
                            success = set_reference_signature(reference_image)
                        if success:
                            st.success(" Reference signature processed!")
                        else:
                            st.error(" Failed to process signature")
            
    
    # Right Column - Verification Signature Section
    with col2:
        # Center the content using columns
        left_spacer2, center_content2, right_spacer2 = st.columns([0.1, 0.8, 0.1])
        
        with center_content2:
            st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
            st.subheader(" Step 2: Verify Signature")
            st.write("Upload ONE signature to verify")
            st.markdown("</div>", unsafe_allow_html=True)
            
            test_file = st.file_uploader(
                "Choose signature to verify",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                key="verification_uploader_main"
            )
            
            # Clear results if verification image is removed
            if test_file is None and hasattr(st.session_state, 'verification_results'):
                del st.session_state.verification_results
            
            if test_file is not None:
                test_image = load_image(test_file)
                if test_image is not None:
                    st.image(test_image, caption="Signature to Verify", use_column_width=True)
                    
                    # Center the image info
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    st.write(f"**Shape:** {test_image.shape}")
                    st.write(f"**Size:** {test_image.shape[0]} x {test_image.shape[1]} pixels")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    if st.session_state.models_trained:
                        if st.button(" Verify Signature", type="primary", use_container_width=True):
                            with st.spinner("Analyzing signature..."):
                                results = compare_signature(test_image)
                                if results:
                                    st.session_state.verification_results = results
                                    st.success("✅ Verification complete!")
                    else:
                        st.warning(" Please process the reference signature first")
    
    # Results Section
    display_verification_results()

def display_verification_results():
    """Display verification results section."""
    if hasattr(st.session_state, 'verification_results') and st.session_state.verification_results:
        results = st.session_state.verification_results
        
        st.markdown("---")
        st.markdown("##  Verification Results")
        
        # Main Result Card
        if results['is_genuine']:
            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, #d4edda 0%, #c3e6cb 100%);
                border: 2px solid #2e8b57;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                text-align: center;
                box-shadow: 0 4px 6px rgba(46, 139, 87, 0.2);
            ">
                <h2 style="color: #155724; margin: 0;"> GENUINE SIGNATURE</h2>
                <h3 style="color: #2e8b57; margin: 10px 0;">Confidence: {results['ensemble_confidence']:.1f}%</h3>
                <p style="color: #155724; margin: 0;">The signature appears to be authentic based on ML analysis</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, #f8d7da 0%, #f5c6cb 100%);
                border: 2px solid #1f77b4;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                text-align: center;
                box-shadow: 0 4px 6px rgba(31, 119, 180, 0.2);
            ">
                <h2 style="color: #721c24; margin: 0;"> FORGED SIGNATURE</h2>
                <h3 style="color: #1f77b4; margin: 10px 0;">Confidence: {results['ensemble_confidence']:.1f}%</h3>
                <p style="color: #721c24; margin: 0;">The signature appears to be forged based on ML analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label=" Final Prediction",
                value=results['ensemble_prediction'],
                delta=None
            )
        
        with col2:
            confidence_color = "normal" if results['ensemble_confidence'] > 70 else "inverse"
            st.metric(
                label=" Confidence Level",
                value=f"{results['ensemble_confidence']:.1f}%",
                delta=None
            )
        
        with col3:
            genuine_count = sum(1 for result in results['algorithms'].values() if result['prediction'] == 'Genuine')
            total_algorithms = len(results['algorithms'])
            st.metric(
                label=" Algorithm Agreement",
                value=f"{genuine_count}/{total_algorithms}",
                delta=f"{(genuine_count/total_algorithms)*100:.0f}% consensus"
            )
        
        with col4:
            similarity = results['similarity_metrics']['cosine_similarity']
            st.metric(
                label=" Similarity Score",
                value=f"{similarity:.3f}",
                delta="Higher = More Similar"
            )
        
        # Detailed Analysis Section
        st.markdown("###  Detailed Analysis")
        
        # Algorithm Results in Professional Table
        st.markdown("#### Individual Algorithm Predictions")
        
        # Create a more professional dataframe
        algo_data = []
        for algo, result in results['algorithms'].items():
            status = " Genuine" if result['prediction'] == 'Genuine' else "❌ Forged"
            confidence = f"{result['confidence']:.1f}%"
            algo_data.append({
                'Algorithm': algo,
                'Prediction': status,
                'Confidence': confidence
            })
        
        algo_df = pd.DataFrame(algo_data)
        st.dataframe(
            algo_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Visual Analysis
        col_chart, col_summary = st.columns([2, 1])
        
        with col_chart:
            create_confidence_visualization(results)
        
        with col_summary:
            st.markdown("####  Analysis Summary")
            
            # Voting breakdown
            st.markdown("**Ensemble Voting:**")
            for algo, weight in results['voting_details']['weights'].items():
                prediction = results['algorithms'][algo]['prediction']
                icon = "" if prediction == 'Genuine' else ""
                st.write(f"{icon} {algo}: {weight:.0%} weight")
            
            st.markdown("---")
            
            # Similarity metrics
            st.markdown("**Similarity Analysis:**")
            cosine_sim = results['similarity_metrics']['cosine_similarity']
            euclidean_dist = results['similarity_metrics']['euclidean_distance']
            
            st.write(f" Cosine Similarity: {cosine_sim:.4f}")
            st.write(f" Euclidean Distance: {euclidean_dist:.4f}")
            
            # Interpretation
            st.markdown("---")
            st.markdown("**Interpretation:**")
            if results['is_genuine']:
                st.success("High similarity to reference signature detected")
            else:
                st.error("Significant differences from reference signature detected")
        
        # Advanced Performance Analytics Section
        st.markdown("---")
        st.markdown("##  Advanced Performance Analytics")
        st.markdown("*Demonstrating Ensemble Method Superiority*")
        
        # Create performance data for visualization
        performance_data = create_performance_analytics(results)
        
        # Row 1: Accuracy Comparison and Precision/Recall/F1
        col_acc, col_metrics = st.columns([1, 1])
        
        with col_acc:
            st.markdown("####  Algorithm Accuracy Comparison")
            create_accuracy_comparison_chart(performance_data)
        
        with col_metrics:
            st.markdown("####  Precision, Recall & F1-Score")
            create_precision_recall_f1_chart(performance_data)
        
        # Row 2: Confusion Matrix and ROC Curves
        col_conf, col_roc = st.columns([1, 1])
        
        with col_conf:
            st.markdown("####  Confusion Matrix Heatmap")
            create_confusion_matrix_heatmap(performance_data)
        
        with col_roc:
            st.markdown("####  ROC Curves Comparison")
            create_roc_curves_chart(performance_data)
        
        # Row 3: Robustness Analysis
        st.markdown("####  Robustness Under Noise Attack")
        create_robustness_analysis_chart(performance_data)

def system_info_tab():
    st.markdown('<div class="section-header">System Information</div>', unsafe_allow_html=True)
    
    # Algorithm information
    st.subheader("ML Algorithms Used")
    
    algorithms_info = {
        "Random Forest": "Ensemble method using multiple decision trees. Combines predictions from many trees for robust classification.",
        "Neural Network": "Multi-layer perceptron with hidden layers. Learns complex patterns through backpropagation training.",
        "Support Vector Machine": "Finds optimal decision boundary using kernel methods. Effective for high-dimensional feature spaces.",
        "Gradient Boosting": "Sequential ensemble method that builds models iteratively. Each new model corrects errors from previous models."
    }
    
    for algo, description in algorithms_info.items():
        with st.expander(f"{algo}"):
            st.write(description)
    
    # Feature extraction info
    st.subheader("Feature Extraction")
    
    st.write("""
    The system extracts 45+ features from each signature image:
    
    **Geometric Features (15):**
    - Area, perimeter, aspect ratio
    - Extent, solidity, compactness
    - Bounding box dimensions
    - Centroid coordinates
    - Hu moments (shape descriptors)
    
    **Texture Features (14):**
    - Local Binary Patterns (LBP)
    - Gray-Level Co-occurrence Matrix (GLCM) properties
    - Contrast, dissimilarity, homogeneity, energy
    
    **Statistical Features (16):**
    - Mean, standard deviation, variance
    - Median, min, max values
    - Skewness and kurtosis
    - Histogram-based features
    """)
    
    # System workflow
    st.subheader("How It Works")
    
    st.write("""
    1. **Reference Setting:** Upload a genuine signature as reference
    2. **Variation Generation:** System creates 20+ slight variations of the reference
    3. **Model Training:** Anomaly detection models learn the reference pattern
    4. **Feature Extraction:** Extract 45+ features from both reference and test signatures
    5. **Multi-Algorithm Analysis:** Apply 5 different ML algorithms for comparison
    6. **Ensemble Decision:** Combine results using weighted voting for final prediction
    """)
    
    # Performance metrics
    if st.session_state.reference_set:
        st.subheader("Current Reference Status")
        
        col1, col2, col3 = st.columns(3)
        
        geometric_count = len([k for k in st.session_state.reference_features.keys() if k.startswith('geo_')])
        texture_count = len([k for k in st.session_state.reference_features.keys() if k.startswith('texture_')])
        statistical_count = len([k for k in st.session_state.reference_features.keys() if k.startswith('stat_')])
        
        with col1:
            st.metric("Total Features", len(st.session_state.reference_features))
        with col2:
            st.metric("Geometric Features", geometric_count)
        with col3:
            st.metric("Texture Features", texture_count)

if __name__ == "__main__":
    main()
