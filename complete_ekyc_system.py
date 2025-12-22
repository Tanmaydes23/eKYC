"""
Complete eKYC System with All 3 Components
Run this in Google Colab to integrate everything

Components:
1. Deepfake Detection - Your trained EfficientNet-B7 ✅
2. Identity Verification - InsightFace ArcFace (pretrained) ✅
3. Liveness Detection - Simple baseline (can upgrade to FeatherNets)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import cv2
import numpy as np
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

print("📦 Installing dependencies...")
os.system('pip install -q insightface onnxruntime-gpu timm facenet-pytorch opencv-python-headless')

import timm
import insightface
from insightface.app import FaceAnalysis
from facenet_pytorch import MTCNN

print("✅ Dependencies installed!")

# =============================================================================
# 1. Load Your Trained Deepfake Model
# =============================================================================
class DeepfakeDetectorEfficientNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnet_b7_ns', pretrained=pretrained, num_classes=0, global_pool='')
        self.feat_dim = self.backbone.num_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.feat_dim, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        pooled = self.global_pool(features).flatten(1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)

class DeepfakeEnsemble(nn.Module):
    def __init__(self, num_models=1):
        super().__init__()
        self.models = nn.ModuleList([DeepfakeDetectorEfficientNet(pretrained=False) for _ in range(num_models)])
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def forward(self, x):
        predictions = [model(x) for model in self.models]
        predictions = torch.stack(predictions, dim=0)
        weights = F.softmax(self.weights, dim=0)
        return torch.sum(predictions * weights.view(-1, 1, 1), dim=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\n1️⃣ Loading your trained deepfake detector...")
deepfake_model = DeepfakeEnsemble(num_models=1).to(device)
checkpoint = torch.load('/content/drive/MyDrive/best_deepfake_model.pth', map_location=device)
deepfake_model.load_state_dict(checkpoint)
deepfake_model.eval()
print("   ✅ Deepfake detector loaded!")

# =============================================================================
# 2. Initialize InsightFace (Identity Verification) - NO TRAINING NEEDED
# =============================================================================
print("\n2️⃣ Initializing InsightFace ArcFace (99.86% LFW accuracy)...")
face_app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
print("   ✅ InsightFace ready!")

# =============================================================================
# 3. Initialize MTCNN for Face Detection
# =============================================================================
print("\n3️⃣ Initializing MTCNN face detector...")
mtcnn = MTCNN(image_size=224, margin=20, device=device, keep_all=False)
print("   ✅ MTCNN ready!")

# =============================================================================
# 4. Simple Liveness Detection (Baseline)
# =============================================================================
def detect_liveness_simple(image):
    """
    Simple liveness detection using texture analysis
    Returns score 0-1 (higher = more likely real)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Calculate Laplacian variance (sharpness)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Normalize to 0-1 range (higher variance = sharper = more likely real)
    liveness_score = min(laplacian_var / 500.0, 1.0)
    
    return liveness_score

# =============================================================================
# Complete eKYC System
# =============================================================================
class CompleteEKYCSystem:
    def __init__(self, deepfake_model, face_app, mtcnn, device):
        self.deepfake_model = deepfake_model
        self.face_app = face_app
        self.mtcnn = mtcnn
        self.device = device
    
    def detect_deepfake(self, image):
        """Detect if image contains deepfake"""
        try:
            # Extract face
            if isinstance(image, str):
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.array(image)
            
            face_tensor = self.mtcnn(img)
            if face_tensor is None:
                return None
            
            # Predict
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.deepfake_model(face_tensor)
                prob = torch.sigmoid(logits).item()
            
            authenticity_score = 1 - prob  # Higher = more authentic
            return authenticity_score
        except Exception as e:
            print(f"Error in deepfake detection: {e}")
            return None
    
    def verify_identity(self, image1, image2):
        """Verify if two images contain the same person"""
        try:
            # Load images
            if isinstance(image1, str):
                img1 = cv2.imread(image1)
            else:
                img1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
            
            if isinstance(image2, str):
                img2 = cv2.imread(image2)
            else:
                img2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
            
            # Get embeddings
            faces1 = self.face_app.get(img1)
            faces2 = self.face_app.get(img2)
            
            if len(faces1) == 0 or len(faces2) == 0:
                return None
            
            emb1 = faces1[0].embedding
            emb2 = faces2[0].embedding
            
            # Calculate similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            match_score = float((similarity + 1) / 2)  # Normalize to 0-1
            
            return match_score
        except Exception as e:
            print(f"Error in identity verification: {e}")
            return None
    
    def detect_liveness(self, image):
        """Detect if image is from a live person"""
        try:
            if isinstance(image, str):
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = image
            
            liveness_score = detect_liveness_simple(img)
            return liveness_score
        except Exception as e:
            print(f"Error in liveness detection: {e}")
            return None
    
    def complete_verification(self, id_image_path, selfie_image_path, 
                            deepfake_threshold=0.5, 
                            identity_threshold=0.6,
                            liveness_threshold=0.3):
        """
        Complete eKYC verification
        
        Args:
            id_image_path: Path to ID photo
            selfie_image_path: Path to selfie photo
            
        Returns:
            dict with all scores and final decision
        """
        print("\n" + "="*80)
        print("🔍 ZenTej AI: Complete eKYC Verification")
        print("="*80)
        
        results = {}
        
        # 1. Deepfake Detection
        print("\n1️⃣ Deepfake Detection...")
        deepfake_score_id = self.detect_deepfake(id_image_path)
        deepfake_score_selfie = self.detect_deepfake(selfie_image_path)
        
        if deepfake_score_id is None or deepfake_score_selfie is None:
            print("   ❌ Face not detected")
            return {'error': 'Face not detected', 'pass': False}
        
        overall_authenticity = min(deepfake_score_id, deepfake_score_selfie)
        results['deepfake_detection'] = {
            'id_authenticity': deepfake_score_id,
            'selfie_authenticity': deepfake_score_selfie,
            'overall_authenticity': overall_authenticity,
            'pass': overall_authenticity > deepfake_threshold
        }
        print(f"   ID Authenticity: {deepfake_score_id:.2%}")
        print(f"   Selfie Authenticity: {deepfake_score_selfie:.2%}")
        print(f"   Overall: {overall_authenticity:.2%} {'✅' if overall_authenticity > deepfake_threshold else '❌'}")
        
        # 2. Identity Verification
        print("\n2️⃣ Identity Verification...")
        identity_score = self.verify_identity(id_image_path, selfie_image_path)
        
        if identity_score is None:
            print("   ❌ Face not detected")
            return {'error': 'Face not detected for verification', 'pass': False}
        
        results['identity_verification'] = {
            'match_score': identity_score,
            'pass': identity_score > identity_threshold
        }
        print(f"   Match Score: {identity_score:.2%} {'✅' if identity_score > identity_threshold else '❌'}")
        
        # 3. Liveness Detection
        print("\n3️⃣ Liveness Detection...")
        liveness_score = self.detect_liveness(selfie_image_path)
        
        results['liveness_detection'] = {
            'liveness_score': liveness_score,
            'pass': liveness_score > liveness_threshold
        }
        print(f"   Liveness Score: {liveness_score:.2%} {'✅' if liveness_score > liveness_threshold else '❌'}")
        
        # Final Decision
        final_pass = (
            results['deepfake_detection']['pass'] and
            results['identity_verification']['pass'] and
            results['liveness_detection']['pass']
        )
        
        results['final_decision'] = {
            'pass': final_pass,
            'confidence': (overall_authenticity + identity_score + liveness_score) / 3
        }
        
        print("\n" + "="*80)
        if final_pass:
            print("✅ ✅ ✅  VERIFICATION PASSED  ✅ ✅ ✅")
        else:
            print("❌ ❌ ❌  VERIFICATION FAILED  ❌ ❌ ❌")
            failed_checks = []
            if not results['deepfake_detection']['pass']:
                failed_checks.append("Deepfake detected")
            if not results['identity_verification']['pass']:
                failed_checks.append("Identity mismatch")
            if not results['liveness_detection']['pass']:
                failed_checks.append("Liveness check failed")
            print(f"Reasons: {', '.join(failed_checks)}")
        print("="*80)
        
        return results

# Initialize complete system
print("\n" + "="*80)
print("🎉 Complete eKYC System Ready!")
print("="*80)
print("Components loaded:")
print("  ✅ Deepfake Detection (Your trained EfficientNet-B7)")
print("  ✅ Identity Verification (InsightFace ArcFace - 99.86% LFW)")
print("  ✅ Liveness Detection (Texture-based baseline)")
print("="*80)

ekyc_system = CompleteEKYCSystem(deepfake_model, face_app, mtcnn, device)

# =============================================================================
# Example Usage
# =============================================================================
print("\n📝 Example Usage:")
print("""
# Verify a person:
results = ekyc_system.complete_verification(
    id_image_path='/path/to/id_photo.jpg',
    selfie_image_path='/path/to/selfie.jpg'
)

# Check result:
print(results['final_decision']['pass'])  # True/False
print(results['final_decision']['confidence'])  # Overall confidence score
""")

print("\n🚀 System ready for testing!")
