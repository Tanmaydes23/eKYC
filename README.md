# eKYC Verification System

**Advanced face verification system with AI-powered fraud detection**

A complete eKYC (electronic Know Your Customer) solution combining three state-of-the-art AI models for comprehensive identity verification.

---

## 🎯 Features

### 3-Layer Security Verification

**1. 🎭 Identity Verification**
- **Model:** InsightFace (buffalo_l)
- **Accuracy:** 99%+
- **Function:** Compares ID photo with selfie to verify same person

**2. 👤 Liveness Detection**
- **Model:** LivenessNet (PyImageSearch)
- **Accuracy:** 90-95%
- **Function:** Detects spoofing attacks (photos, videos, masks)

**3. 🤖 Deepfake Detection**
- **Model:** Vision Transformer (ViT) - dima806/deepfake_vs_real_image_detection
- **Accuracy:** 90-95%
- **Function:** Identifies AI-generated or manipulated faces

---

## � Quick Start

### Prerequisites
- Python 3.8+
- Anaconda or virtualenv
- 4GB RAM minimum
- (Optional) GPU for faster processing

### Installation

1. **Clone or navigate to directory:**
```bash
cd C:\Kaggle
```

2. **Install dependencies:**
```bash
pip install streamlit insightface opencv-python-headless numpy pillow
pip install transformers tf-keras tensorflow torch
```

3. **Run the webapp:**
```bash
streamlit run ekyc_webapp.py
```

4. **Open browser:**
The app will automatically open at `http://localhost:8501`

---

## 📁 Project Structure

```
C:\Kaggle\
├── ekyc_webapp.py              # Main Streamlit application
├── liveness_net.py             # LivenessNet detector module
├── liveness.h5                 # LivenessNet model weights (1.8 MB)
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
├── Duplication_Dataset/        # Real face images (5,000)
└── Forgery_Dataset/            # Deepfake images (20,000)
```

**Models downloaded automatically on first run:**
- InsightFace buffalo_l (~200 MB)
- ViT deepfake detector (~343 MB)

---

## 🎮 Usage

### Web Interface

1. **Upload ID Photo** - Government ID or passport photo
2. **Upload Selfie** - Recent photo of the person
3. **Click "Verify eKYC"**
4. **Review Results:**
   - ✅ **PASS** - All checks passed
   - ⚠️ **UNCERTAIN** - Manual review recommended
   - ❌ **FAIL** - Verification failed

### Verification Thresholds

Default thresholds (adjustable in sidebar):
- **Deepfake Detection:** 70% authenticity required
- **Identity Match:** 50% similarity required
- **Liveness Check:** 50% liveness required

---

## � Technical Details

### Architecture

**Identity Verification (InsightFace):**
- ArcFace loss function
- 512-dimensional face embeddings
- Cosine similarity matching

**Liveness Detection (LivenessNet):**
- Binary CNN classifier
- Input: 32x32 RGB face crops
- Output: Real vs Spoof probability

**Deepfake Detection (ViT):**
- Vision Transformer architecture
- Pre-trained on deepfake datasets
- Fine-tuned for generalization

### Performance

| Component | Model | Input Size | Inference Time |
|-----------|-------|------------|----------------|
| Identity | InsightFace | 640x640 | ~0.5s |
| Liveness | LivenessNet | 32x32 | ~0.1s |
| Deepfake | ViT | 224x224 | ~0.8s |

**Total verification time:** < 2 seconds per user

---

## 📊 Model Information

### InsightFace
- **License:** Academic use
- **Source:** [InsightFace GitHub](https://github.com/deepinsight/insightface)
- **Auto-downloaded:** Yes

### LivenessNet
- **License:** Custom trained model
- **Architecture:** Binary CNN (PyImageSearch)
- **File:** `liveness.h5` (included)

### ViT Deepfake Detector
- **License:** Apache 2.0
- **Source:** [Hugging Face - dima806](https://huggingface.co/dima806/deepfake_vs_real_image_detection)
- **Auto-downloaded:** Yes
- **Training:** 99.27% accuracy on original dataset

---

## 🛠️ Configuration

### Adjusting Thresholds

Open the webapp sidebar to adjust:
```python
deepfake_threshold = 0.7    # 70% authenticity required
identity_threshold = 0.5    # 50% similarity required
liveness_threshold = 0.5    # 50% liveness required
```

### Model Paths

Models are automatically managed:
- InsightFace: `~/.insightface/models/`
- ViT: `~/.cache/huggingface/`
- LivenessNet: `./liveness.h5`

---

## 🐛 Troubleshooting

### Common Issues

**1. "Failed to load liveness model"**
- Ensure `liveness.h5` exists in project directory
- File size should be ~1.8 MB

**2. "InsightFace model download failed"**
- Check internet connection
- May require VPN in restricted regions

**3. "ViT model loading slowly"**
- First run downloads 343 MB
- Subsequent runs are instant (cached)

**4. Low GPU memory**
- Models default to CPU
- All inference works on CPU (slower but functional)

---

## 📈 Future Improvements

- [ ] Video liveness detection
- [ ] Multi-face handling
- [ ] API endpoint integration
- [ ] Mobile app support
- [ ] Database integration
- [ ] Batch processing

---

## � License

**Project:** Custom eKYC implementation

**Model Licenses:**
- InsightFace: Academic use
- LivenessNet: Custom trained
- ViT (dima806): Apache 2.0

---



## 🙏 Acknowledgments

- **InsightFace** - Face recognition backbone
- **PyImageSearch** - LivenessNet architecture
- **dima806** - Pre-trained ViT deepfake model
- **Hugging Face** - Model hosting and transformers library

---

## 📧 Support

For issues or questions:
1. Check this README
2. Review `ekyc_webapp.py` comments
3. Verify model files are present

---

