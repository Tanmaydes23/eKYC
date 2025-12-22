"""
ZenTej AI: Deepfake-Proof eKYC - Training Script (COLAB ONLY!)
Competition-Winning Solution using SOTA Models

⚠️ THIS SCRIPT ONLY RUNS IN GOOGLE COLAB ⚠️

USAGE:
1. Open this file in VS Code
2. Press Shift+Enter (or right-click → Run in Interactive Window)
3. Select "Colab" runtime with T4 GPU
4. The script will automatically handle everything!

DO NOT run locally - use train_ekyc.py for local training instead.
"""

import sys
import os

# ============================================================================
# COLAB-ONLY ENFORCEMENT
# ============================================================================
def check_colab_or_exit():
    """Ensure this script only runs in Google Colab"""
    try:
        import google.colab
        return True
    except:
        print("\n" + "="*80)
        print("❌ ERROR: This script can ONLY run in Google Colab!")
        print("="*80)
        print("\n📍 You are trying to run locally.")
        print("\n✅ To use Google Colab T4 GPU:")
        print("   1. Open this file in VS Code")
        print("   2. Press Shift+Enter")
        print("   3. Select 'Colab' runtime (NOT Python)")
        print("   4. Choose T4 GPU")
        print("\n💡 For local training, use: train_ekyc.py instead")
        print("="*80 + "\n")
        sys.exit(1)

# Check immediately
check_colab_or_exit()
print("✅ Running in Google Colab - proceeding with setup...")

import sys
import os
from pathlib import Path

# ============================================================================
# MOUNT GOOGLE DRIVE (Authorization happens here)
# ============================================================================
print("\n" + "="*80)
print("📁 Mounting Google Drive - Authorization Required")
print("="*80)
print("\n💡 IMPORTANT: Look for the authorization URL in the output!")
print("   1. Find the URL starting with 'https://accounts.google.com...'")
print("   2. Copy and open it in your browser")
print("   3. Sign in and copy the authorization code")
print("   4. Paste the code back here when prompted\n")

from google.colab import drive
drive.mount('/content/drive', force_remount=False)
print("✅ Google Drive mounted successfully!\n")

# ============================================================================
# SETUP DATASET
# ============================================================================
print("🚀 Setting up dataset...")

# Check for zipped dataset
zip_path = Path('/content/drive/MyDrive/Kaggle.zip')
extract_path = Path('/content/kaggle_data')

if zip_path.exists():
    print(f"📦 Found Kaggle.zip in Drive")
    print(f"📂 Extracting to {extract_path}...")
    
    import zipfile
    import shutil
    
    if extract_path.exists():
        print("   🗑️  Removing old extracted data...")
        shutil.rmtree(extract_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        total_files = len(zip_ref.namelist())
        print(f"   📊 Extracting {total_files} files...")
        zip_ref.extractall(extract_path)
    
    print("✅ Dataset extracted!")
    
    # Find dataset root
    kaggle_dirs = list(extract_path.glob('**/Kaggle'))
    if kaggle_dirs:
        DATA_ROOT = kaggle_dirs[0]
    elif (extract_path / 'Duplication_Dataset').exists():
        DATA_ROOT = extract_path
    else:
        DATA_ROOT = extract_path
    
    print(f"✅ Dataset root: {DATA_ROOT}")
else:
    print(f"\n❌ ERROR: Kaggle.zip not found!")
    print(f"   Expected: /content/drive/MyDrive/Kaggle.zip")
    print(f"\n💡 Upload Kaggle.zip to Google Drive root (MyDrive folder)")
    sys.exit(1)

# Setup Colab environment
def setup_colab_environment():
    """Legacy function - kept for compatibility"""
    return DATA_ROOT

DATA_ROOT = setup_colab_environment()

# Install dependencies
print("\n📦 Installing Python dependencies...")
print("   Fixing Pillow compatibility issue...")

# Uninstall problematic Pillow version and reinstall compatible version
os.system('pip uninstall -y Pillow')
os.system('pip install -q Pillow==10.0.1')  # Compatible with torchvision

print("   Installing deep learning packages...")
os.system('pip install -q onnxruntime-gpu timm insightface facenet-pytorch albumentations opencv-python-headless scikit-learn tqdm')

print("\n⚠️  IMPORTANT: Runtime restart required for Pillow fix!")
print("Please go to: Runtime -> Restart runtime")
print("Then run this cell again.\n")

# Check if this is the first run (Pillow just installed)
try:
    from PIL import _util
    if not hasattr(_util, 'is_directory'):
        print("❌ Old Pillow detected. Please restart runtime and run again!")
        import sys
        sys.exit(0)
except:
    pass

print("✅ All dependencies installed!")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import insightface
from facenet_pytorch import MTCNN
import cv2
import numpy as np
from tqdm.auto import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Paths - Auto-configured based on environment
    DUPLICATION_DIR = DATA_ROOT / 'Duplication_Dataset'
    FORGERY_DIR = DATA_ROOT / 'Forgery_Dataset'
    
    # Training
    BATCH_SIZE = 4  # Reduced for T4 GPU (15GB VRAM)
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 2  # Quick test run (1 hour) - increase later
    WEIGHT_DECAY = 1e-5
    
    # Model
    NUM_MODELS = 1  # Start with 1 for T4 GPU, can increase later
    IMAGE_SIZE = 224
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = Config.DEVICE
print(f'\n🚀 Using device: {device}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

# ============================================================================
# FACE DETECTION
# ============================================================================
print('\n🔍 Initializing MTCNN Face Detector...')
mtcnn = MTCNN(
    image_size=Config.IMAGE_SIZE,
    margin=20,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device,
    keep_all=False
)
print('✅ MTCNN initialized')

def extract_face_from_image(image_path, mtcnn):
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(img_rgb)
        return face
    except:
        return None

# ============================================================================
# DATASET
# ============================================================================
class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, mtcnn, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.mtcnn = mtcnn
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        face = extract_face_from_image(img_path, self.mtcnn)
        if face is None:
            face = torch.zeros(3, 224, 224)
        
        if self.transforms:
            face_np = face.permute(1, 2, 0).numpy()
            augmented = self.transforms(image=face_np)
            face = torch.from_numpy(augmented['image']).permute(2, 0, 1)
        
        return face, torch.tensor(label, dtype=torch.float32)

# Augmentations
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.OneOf([A.GaussNoise(p=1), A.GaussianBlur(p=1), A.MedianBlur(blur_limit=5, p=1)], p=0.5),
    A.ImageCompression(quality_lower=70, quality_upper=100, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
    A.Resize(224, 224),
])

val_transforms = A.Compose([A.Resize(224, 224)])

# ============================================================================
# MODEL
# ============================================================================
class DeepfakeDetectorEfficientNet(nn.Module):
    def __init__(self, pretrained=True):
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
    def __init__(self, num_models=3):
        super().__init__()
        self.models = nn.ModuleList([DeepfakeDetectorEfficientNet(pretrained=True) for _ in range(num_models)])
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def forward(self, x):
        predictions = [model(x) for model in self.models]
        predictions = torch.stack(predictions, dim=0)
        weights = F.softmax(self.weights, dim=0)
        return torch.sum(predictions * weights.view(-1, 1, 1), dim=0)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.unsqueeze(1), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions.squeeze() == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    return running_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predictions = torch.sigmoid(outputs)
            correct += ((predictions > 0.5).float().squeeze() == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    auc = roc_auc_score(all_labels, all_preds)
    return running_loss / len(dataloader), correct / total, auc

def save_checkpoint(epoch, model, optimizer, scheduler, best_auc, history, filename='checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_auc': best_auc,
        'history': history,
    }
    # Save to Colab workspace
    torch.save(checkpoint, filename)
    
    # Also save to Google Drive for persistence
    drive_path = '/content/drive/MyDrive/' + filename
    torch.save(checkpoint, drive_path)
    print(f'   💾 Checkpoint saved to Drive: {drive_path}')

def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth'):
    # Try loading from Drive first (persists across sessions)
    drive_path = '/content/drive/MyDrive/' + filename
    
    if Path(drive_path).exists():
        filename = drive_path
        print(f'   📂 Loading checkpoint from Drive: {filename}...')
    elif Path(filename).exists():
        print(f'   📂 Loading checkpoint from workspace: {filename}...')
    else:
        print(f'   ℹ️  No checkpoint found, starting from scratch')
        return 0, 0.0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    
    checkpoint = torch.load(filename, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f'   ✅ Resumed from epoch {checkpoint["epoch"]} (best AUC: {checkpoint["best_auc"]:.4f})')
    
    return checkpoint['epoch'] + 1, checkpoint['best_auc'], checkpoint['history']

# ============================================================================
# MAIN
# ============================================================================
def main():
    print('='*80)
    print('🏆 ZenTej AI: Deepfake-Proof eKYC Training')
    print('='*80)
    
    # Gather dataset
    print('\n📁 Collecting dataset files...')
    img_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    video_exts = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    
    duplication_files = []
    for ext in img_exts + video_exts:
        duplication_files.extend(list(Config.DUPLICATION_DIR.glob(f'**/{ext}')))
    
    forgery_files = []
    for ext in img_exts + video_exts:
        forgery_files.extend(list(Config.FORGERY_DIR.glob(f'**/{ext}')))
    
    print(f'   Duplication files: {len(duplication_files)}')
    print(f'   Forgery files: {len(forgery_files)}')
    
    all_files = duplication_files + forgery_files
    all_labels = [0] * len(duplication_files) + [1] * len(forgery_files)
    
    # Train/Val split
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    print(f'\n✂️ Train/Val Split:')
    print(f'   Training: {len(train_files)} samples')
    print(f'   Validation: {len(val_files)} samples')
    
    # Create datasets
    train_dataset = DeepfakeDataset(train_files, train_labels, mtcnn, train_transforms)
    val_dataset = DeepfakeDataset(val_files, val_labels, mtcnn, val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # Create model
    print('\n🔨 Building EfficientNet-B7 Ensemble...')
    model = DeepfakeEnsemble(num_models=Config.NUM_MODELS).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'✅ Model created ({total_params:,} parameters, ~{total_params * 4 / 1e6:.1f} MB)')
    
    # Training setup
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)
    
    # Load checkpoint
    print('\n🔄 Checking for checkpoint...')
    start_epoch, best_auc, history = load_checkpoint(model, optimizer, scheduler)
    
    # Training loop
    print(f'\n🚀 {"Resuming" if start_epoch > 0 else "Starting"} training...')
    print('-'*80)
    
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        print(f'\n📅 Epoch {epoch+1}/{Config.NUM_EPOCHS} (LR: {optimizer.param_groups[0]["lr"]:.2e})')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        print(f'\n📊 Results: Train Loss={train_loss:.4f} Acc={train_acc*100:.2f}% | Val Loss={val_loss:.4f} Acc={val_acc*100:.2f}% AUC={val_auc:.4f}')
        
        save_checkpoint(epoch, model, optimizer, scheduler, best_auc, history)
        
        if val_auc > best_auc:
            best_auc = val_auc
            # Save to both locations
            torch.save(model.state_dict(), 'best_deepfake_model.pth')
            torch.save(model.state_dict(), '/content/drive/MyDrive/best_deepfake_model.pth')
            print(f'   ⭐ New best model! (AUC: {val_auc:.4f}) - Saved to Drive')
    
    print('\n' + '='*80)
    print(f'🎉 Training Complete! Best AUC: {best_auc:.4f}')
    print('='*80)

if __name__ == '__main__':
    main()
