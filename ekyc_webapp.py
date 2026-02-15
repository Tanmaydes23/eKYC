"""
Complete eKYC System - Streamlit Web Demo
Interactive web interface for eKYC verification with all 3 components

Run: streamlit run ekyc_webapp.py
"""

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import insightface
from insightface.app import FaceAnalysis
from transformers import AutoImageProcessor, AutoModelForImageClassification
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import tempfile
from liveness_net import create_liveness_detector

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="ZenTej AI - eKYC Verification",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .fail-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-card {
        padding: 15px;
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Model Definitions (Removed EfficientNet, using ViT from HuggingFace)
# =============================================================================

# =============================================================================
# Load Models (Cached)
# =============================================================================
@st.cache_resource
def load_deepfake_model():
    """Load ViT deepfake detection model"""
    try:
        st.info("📥 Loading ViT Deepfake Detector (dima806)...")
        model_name = "dima806/deepfake_vs_real_image_detection"
        
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)
        model.eval()
        
        st.success("✅ ViT Deepfake Detector loaded successfully!")
        return processor, model
    except Exception as e:
        st.error(f"❌ Failed to load ViT model: {e}")
        return None, None

@st.cache_resource
def load_identity_model():
    """Load InsightFace for identity verification"""
    with st.spinner("🔄 Loading InsightFace model..."):
        try:
            face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
            st.success("✅ InsightFace loaded successfully!")
            return face_app
        except Exception as e:
            st.error(f"❌ CRITICAL: Failed to load InsightFace: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None

@st.cache_resource
def load_liveness_model():
    """Load LivenessNet Detector"""
    with st.spinner("🔄 Loading LivenessNet model..."):
        liveness_model_path = 'liveness.h5'  # LivenessNet Keras model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            liveness_detector = create_liveness_detector(model_path=liveness_model_path, device=device)
            st.success("✅ LivenessNet loaded successfully!")
            return liveness_detector
        except Exception as e:
            st.error(f"❌ CRITICAL: Failed to load LivenessNet: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
            import traceback
            st.code(traceback.format_exc())
            return None  # Set to None instead of crashing

# =============================================================================
# Liveness Detection
# =============================================================================
# from liveness_net import create_liveness_detector # This import is already at the top

def detect_liveness(image, liveness_detector, face_app):
    """
    Advanced liveness detection using MiniFASNet V2 or improved texture analysis
    
    Args:
        image: PIL Image
        liveness_detector: LivenessDetector instance from liveness_detection.py
        face_app: InsightFace FaceAnalysis app (to extract face region)
    
    Returns:
        float: liveness score (0-1, higher = more likely live)
    """
    
    try:
        # Safety check first
        if liveness_detector is None:
            st.error("❌ Liveness detector not initialized!")
            return 0.5  # Neutral score
        
        # Convert to numpy
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Get face bounding box using InsightFace
        faces = face_app.get(img_bgr)
        
        if not faces:
            # If no face detected, use whole image
            st.warning("⚠️ No face detected by InsightFace for liveness, using full image")
            face_region = img_bgr
        else:
            # Get the first face bounding box
            box = faces[0].bbox
            x1, y1, x2, y2 = box
            
            # Calculate center and size
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # CRITICAL: Apply 2.7x scale expansion (matches model training)
            # Model: 2.7_80x80_MiniFASNetV2.pth
            scale = 2.7
            new_width = width * scale
            new_height = height * scale
            
            # Calculate expanded bounding box
            exp_x1 = int(center_x - new_width / 2)
            exp_y1 = int(center_y - new_height / 2)
            exp_x2 = int(center_x + new_width / 2)
            exp_y2 = int(center_y + new_height / 2)
            
            # Clamp to image boundaries
            h, w = img_bgr.shape[:2]
            exp_x1 = max(0, exp_x1)
            exp_y1 = max(0, exp_y1)
            exp_x2 = min(w, exp_x2)
            exp_y2 = min(h, exp_y2)
            
            # Crop face region with 2.7x expansion
            face_region = img_bgr[exp_y1:exp_y2, exp_x1:exp_x2]
            
            if face_region.size == 0:
                st.warning("⚠️ Face crop resulted in empty region, using full image")
                face_region = img_bgr
        
        # Predict liveness
        liveness_score = liveness_detector.predict(face_region)
        
        # Validate score
        if not (0 <= liveness_score <= 1):
            st.warning(f"⚠️ Unusual liveness score: {liveness_score:.4f}, clamping to [0,1]")
            liveness_score = max(0, min(1, liveness_score))
        
        return liveness_score
        
    except Exception as e:
        st.error(f"❌ Error in liveness detection: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        # Fallback to very basic texture check
        try:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            fallback_score = min(variance / 1000.0, 1.0)
            st.info(f"🔄 Using fallback texture analysis, score: {fallback_score:.4f}")
            return fallback_score
        except:
            return 0.5  # Ultimate fallback

# =============================================================================
# Verification Functions
# =============================================================================
def detect_deepfake(image, processor, model):
    """Detect if image contains deepfake using ViT model"""
    try:
        # Preprocess image
        inputs = processor(images=image, return_tensors="pt")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
        # Labels: 0 = Real, 1 = Fake (verified from model config)
        real_prob = probabilities[0, 0].item()  # Get REAL probability
        
        return real_prob
    except Exception as e:
        st.error(f"Error in deepfake detection: {e}")
        return 0.5  # Neutral score on error

def verify_identity(image1, image2, face_app):
    """Verify if two images contain the same person"""
    try:
        img1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
        
        faces1 = face_app.get(img1)
        faces2 = face_app.get(img2)
        
        if len(faces1) == 0 or len(faces2) == 0:
            return None
        
        emb1 = faces1[0].embedding
        emb2 = faces2[0].embedding
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        match_score = float((similarity + 1) / 2)
        
        return match_score
    except Exception as e:
        st.error(f"Error in identity verification: {e}")
        return None

# =============================================================================
# Main App
# =============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🔐 ZenTej AI - eKYC Verification</h1>', unsafe_allow_html=True)
    st.markdown("### Complete 3-Component Identity Verification System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model upload (removed deepfake model upload as it's from HuggingFace)
        # The liveness model is still a local file, so we could add an uploader for it if needed.
        # For now, assuming 'liveness.model' exists.
        
        st.markdown("---")
        
        # Thresholds
        st.subheader("🎚️ Verification Thresholds")
        deepfake_threshold = st.slider("Deepfake Threshold (Authenticity)", 0.0, 1.0, 0.7, 0.05,
                                       help="Higher = stricter deepfake detection (score > threshold means REAL)")
        identity_threshold = st.slider("Identity Match Threshold", 0.0, 1.0, 0.5, 0.05,
                                      help="Higher = stricter identity matching")
        liveness_threshold = st.slider("Liveness Threshold", 0.0, 1.0, 0.5, 0.05,
                                      help="Higher = stricter liveness detection")
        
        st.markdown("---")
        
        # Info
        st.subheader("ℹ️ System Info")
        device_info = "🖥️ GPU (CUDA)" if torch.cuda.is_available() else "💻 CPU"
        st.info(f"**Device:** {device_info}")
        st.info("**Components:**\n- Deepfake Detection\n- Identity Verification\n- Liveness Detection")
    
    # Load models
    st.sidebar.header("🔧 System Status")
    
    # Identity model
    face_app = load_identity_model()
    if face_app is None:
        st.error("❌ Identity verification model failed to load. Please check logs.")
        return
    
    # Liveness model  
    liveness_detector = load_liveness_model()
    if liveness_detector is None:
        st.error("❌ Liveness detection model failed to load. Please check logs.")
        return
    
    # Deepfake model (ViT)
    deepfake_processor, deepfake_model = load_deepfake_model()
    if deepfake_processor is None or deepfake_model is None:
        st.error("❌ Deepfake detection model failed to load. Please check logs.")
        return
    
    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 ID Document Photo")
        id_image = st.file_uploader("Upload ID photo", type=['jpg', 'jpeg', 'png'], key='id')
        if id_image:
            id_img = Image.open(id_image).convert('RGB')
            st.image(id_img, caption="ID Photo", use_column_width=True)
    
    with col2:
        st.subheader("🤳 Selfie Photo")
        selfie_image = st.file_uploader("Upload selfie", type=['jpg', 'jpeg', 'png'], key='selfie')
        if selfie_image:
            selfie_img = Image.open(selfie_image).convert('RGB')
            st.image(selfie_img, caption="Selfie", use_column_width=True)
    
    st.markdown("---")
    
    # Verify button
    if st.button("🔍 Verify Identity", type="primary"):
        if not id_image or not selfie_image:
            st.error("❌ Please upload both ID and selfie images!")
            return
        
        # Run verification
        with st.spinner("🔄 Running verification..."):
            # 1. Deepfake Detection
            progress = st.progress(0)
            st.info("1️⃣ Checking for deepfakes...")
            
            # Deepfake detection is typically applied to the selfie
            deepfake_selfie_score = detect_deepfake(selfie_img, deepfake_processor, deepfake_model)
            
            if deepfake_selfie_score is None:
                st.error("❌ Deepfake detection failed for selfie!")
                return
            
            deepfake_pass = deepfake_selfie_score > deepfake_threshold
            
            progress.progress(33)
            
            # 2. Identity Verification
            st.info("2️⃣ Verifying identity match...")
            identity_score = verify_identity(id_img, selfie_img, face_app)
            
            if identity_score is None:
                st.error("❌ Could not extract face features from one or both images for identity verification!")
                return
            
            identity_pass = identity_score > identity_threshold
            
            progress.progress(66)
            
            # 3. Liveness Detection
            st.info("3️⃣ Checking liveness...")
            liveness_score = detect_liveness(selfie_img, liveness_detector, face_app)
            liveness_pass = liveness_score > liveness_threshold
            
            progress.progress(100)
        
        st.markdown("---")
        
        # Results
        st.header("📊 Verification Results")
        
        # Individual component results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Deepfake Detection", f"{deepfake_selfie_score:.1%}", 
                     delta="✅ Pass" if deepfake_pass else "❌ Fail")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Identity Match", f"{identity_score:.1%}",
                     delta="✅ Pass" if identity_pass else "❌ Fail")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Liveness Check", f"{liveness_score:.1%}",
                     delta="✅ Pass" if liveness_pass else "❌ Fail")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Final decision
        final_pass = deepfake_pass and identity_pass and liveness_pass
        confidence = (deepfake_selfie_score + identity_score + liveness_score) / 3
        
        if final_pass:
            st.markdown(f'<div class="success-box">✅ VERIFICATION PASSED<br>Confidence: {confidence:.1%}</div>', 
                       unsafe_allow_html=True)
            st.balloons()
        else:
            failed_checks = []
            if not deepfake_pass:
                failed_checks.append("❌ Deepfake detected")
            if not identity_pass:
                failed_checks.append("❌ Identity mismatch")
            if not liveness_pass:
                failed_checks.append("❌ Liveness check failed")
            
            st.markdown(f'<div class="fail-box">❌ VERIFICATION FAILED<br>Confidence: {confidence:.1%}</div>', 
                       unsafe_allow_html=True)
            st.error("**Failed checks:**\n" + "\n".join(failed_checks))
        
        # Detailed scores
        with st.expander("📈 Detailed Scores"):
            st.json({
                "Deepfake Detection": {
                    "Selfie Authenticity": f"{deepfake_selfie_score:.4f}",
                    "Threshold": deepfake_threshold,
                    "Pass": deepfake_pass
                },
                "Identity Verification": {
                    "Match Score": f"{identity_score:.4f}",
                    "Threshold": identity_threshold,
                    "Pass": identity_pass
                },
                "Liveness Detection": {
                    "Liveness Score": f"{liveness_score:.4f}",
                    "Threshold": liveness_threshold,
                    "Pass": liveness_pass
                },
                "Final Decision": {
                    "Pass": final_pass,
                    "Overall Confidence": f"{confidence:.4f}"
                }
            })

if __name__ == "__main__":
    # Initialize session state
    if 'model_path' not in st.session_state:
        st.session_state['model_path'] = 'best_deepfake_model.pth'
    
    main()
