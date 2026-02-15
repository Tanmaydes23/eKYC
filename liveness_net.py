import numpy as np
import cv2
from tf_keras.models import load_model
from tf_keras.preprocessing.image import img_to_array

class LivenessNetDetector:
    """
    Binary liveness classifier using LivenessNet (PyImageSearch)
    Classifies faces as Real/Live or Fake/Spoof
    """
    
    def __init__(self, model_path='liveness.model'):
        """
        Initialize LivenessNet detector
        
        Args:
            model_path: path to trained Keras model file
        """
        print(f"Loading LivenessNet from {model_path}...")
        self.model = load_model(model_path)
        self.input_size = (32, 32)  # LivenessNet uses 32x32 input
        print(f"✅ LivenessNet loaded successfully")
        print(f"   Input shape: {self.model.input_shape}")
        print(f"   Output shape: {self.model.output_shape}")
    
    def predict(self, face_img, debug=False):
        """
        Predict liveness score for a face image
        
        Args:
            face_img: numpy array (RGB), face region
            debug: if True, print debugging information
        
        Returns:
            float: liveness score (0-1, higher = more likely live)
        """
        if debug:
            print(f"[DEBUG] Input shape: {face_img.shape}")
            print(f"[DEBUG] Input dtype: {face_img.dtype}")
            print(f"[DEBUG] Input range: [{face_img.min()}, {face_img.max()}]")
        
        # Preprocess
        # 1. Resize to 32x32
        face_resized = cv2.resize(face_img, self.input_size)
        
        # 2. Convert to float and normalize to [0, 1]
        face_array = img_to_array(face_resized)
        face_array = face_array.astype("float") / 255.0
        
        # 3. Expand dimensions for batch
        face_array = np.expand_dims(face_array, axis=0)
        
        if debug:
            print(f"[DEBUG] After preprocess shape: {face_array.shape}")
            print(f"[DEBUG] After preprocess range: [{face_array.min():.3f}, {face_array.max():.3f}]")
        
        # Predict
        preds = self.model.predict(face_array, verbose=0)[0]
        
        if debug:
            print(f"[DEBUG] Raw predictions: {preds}")
            print(f"[DEBUG] Fake probability: {preds[0]:.6f}")
            print(f"[DEBUG] Real probability: {preds[1]:.6f}")
        
        # preds[0] = Fake, preds[1] = Real
        # Return Real probability as liveness score
        liveness_score = preds[1]
        
        if debug:
            print(f"[DEBUG] Final liveness score: {liveness_score:.6f}")
            print(f"[DEBUG] Classification: {'LIVE ✅' if liveness_score > 0.5 else 'FAKE ❌'}")
        
        return float(liveness_score)


def create_liveness_detector(model_path='liveness.model', device=None):
    """
    Factory function to create LivenessNet detector
    
    Args:
        model_path: path to model file
        device: unused (for API compatibility)
    
    Returns:
        LivenessNetDetector instance
    """
    return LivenessNetDetector(model_path)
