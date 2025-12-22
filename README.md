# Deepfake-Resistant eKYC Verification System

![AI](https://img.shields.io/badge/AI-Computer%20Vision-blue)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ML-red)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-green)


**An end-to-end AI system for secure eKYC verification with deepfake detection, liveness checks, and identity matching**

---

## 📌 Overview

This project implements a **deepfake-resistant eKYC verification pipeline** designed for secure digital identity onboarding.  
The system processes facial images and performs **multi-stage verification** to assess authenticity and identity.

The project focuses on **ML system integration, and real-time inference**

---

## ✨ Key Features

- 👤 Face detection and alignment from images
- 🧠 Deepfake authenticity detection
- 🎥 Liveness and anti-spoofing checks
- 🔐 Facial identity verification using embedding-based matching
- 📊 Confidence-based decision outputs
- 🌐 Interactive web interface for real-time verification

---

## 🏗️ High-Level Architecture

The system follows a **four-stage verification pipeline**:

1. **Face Detection** – Detect and align faces from input media  
2. **Deepfake Detection** – Classify media as real or manipulated  
3. **Liveness Detection** – Detect spoofing or presentation attacks  
4. **Identity Verification** – Match facial embeddings for identity confirmation  

> The pipeline is modular and designed for clarity, robustness, and extensibility.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- PyTorch
- Webcam or image/video input

### Backend / Model Inference
```bash
cd backend
pip install -r requirements.txt
python main.py
```
### Frontend
``` bash
cd app
streamlit run app.py



