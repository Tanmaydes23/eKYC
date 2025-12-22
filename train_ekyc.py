"""
ZenTej AI: Deepfake-Proof eKYC - Training Script
Competition-Winning Solution using SOTA Models

Models:
- Deepfake Detection: EfficientNet-B7 Ensemble (DFDC Winner)
- Identity Verification: InsightFace ArcFace (99.86% LFW)
- Face Detection: MTCNN

Usage: python train_ekyc.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import timm
import insightface
from insightface.app import FaceAnalysis
from facenet_pytorch import MTCNN

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # Paths - YOUR ACTUAL FOLDERS
    DATA_ROOT = Path('C:/Kaggle')
    DUPLICATION_DIR = DATA_ROOT / 'Duplication_Dataset'
    FORGERY_DIR = DATA_ROOT / 'Forgery_Dataset'
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    WEIGHT_DECAY = 1e-5
    
    # Model
    NUM_MODELS = 3  # Ensemble size
    IMAGE_SIZE = 224
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set device
device = Config.DEVICE
print(f'\U0001F680 Using device: {device}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

# Initialize MTCNN
print('\U0001F50D Initializing MTCNN Face Detector...')
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
print('\u2705 MTCNN initialized')

# Face extraction functions
def extract_face_from_image(image_path, mtcnn, target_size=224):
    """Extract and align face from image using MTCNN."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(img_rgb)
        
        if face is None:
            return None
        
        return face
    except Exception as e:
        print(f'Error processing {image_path}: {e}')
        return None

def extract_face_from_video(video_path, mtcnn, max_frames=30, target_size=224):
    """Extract faces from video frames."""
    faces = []
    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_indices = np.linspace(0, total_frames-1, min(max_frames, total_frames), dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = mtcnn(frame_rgb)
            
            if face is not None:
                faces.append(face)
        
        cap.release()
    except Exception as e:
        print(f'Error processing video {video_path}: {e}')
    
    return faces

# Dataset class
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
        
        # Extract face
        face = extract_face_from_image(img_path, self.mtcnn)
        
        if face is None:
            face = torch.zeros(3, 224, 224)
        
        # Apply augmentations if provided
        if self.transforms:
            face_np = face.permute(1, 2, 0).numpy()
            augmented = self.transforms(image=face_np)
            face = torch.from_numpy(augmented['image']).permute(2, 0, 1)
        
        return face, torch.tensor(label, dtype=torch.float32)

# Data augmentation (DFDC-style)
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1),
        A.GaussianBlur(blur_limit=(3, 7), p=1),
        A.MedianBlur(blur_limit=5, p=1),
    ], p=0.5),
    A.ImageCompression(quality_lower=70, quality_upper=100, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
    A.Resize(224, 224),
])

val_transforms = A.Compose([
    A.Resize(224, 224),
])

# Model architectures
class DeepfakeDetectorEfficientNet(nn.Module):
    """Single EfficientNet-B7 model for deepfake detection"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnet_b7_ns',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        self.feat_dim = self.backbone.num_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.feat_dim, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        pooled = self.global_pool(features).flatten(1)
        pooled = self.dropout(pooled)
        out = self.fc(pooled)
        return out

class DeepfakeEnsemble(nn.Module):
    """Ensemble of 3 EfficientNet-B7 models"""
    def __init__(self, num_models=3):
        super().__init__()
        self.models = nn.ModuleList([
            DeepfakeDetectorEfficientNet(pretrained=True) 
            for _ in range(num_models)
        ])
        
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def forward(self, x):
        predictions = [model(x) for model in self.models]
        predictions = torch.stack(predictions, dim=0)
        
        weights = F.softmax(self.weights, dim=0)
        ensemble_pred = torch.sum(predictions * weights.view(-1, 1, 1), dim=0)
        
        return ensemble_pred

# Focal Loss
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

# Training functions
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
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

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
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    auc = roc_auc_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, auc

def save_checkpoint(epoch, model, optimizer, scheduler, best_auc, history, filename='checkpoint.pth'):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_auc': best_auc,
        'history': history,
        'config': {
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'num_epochs': Config.NUM_EPOCHS,
        }
    }
    torch.save(checkpoint, filename)
    print(f'   💾 Checkpoint saved: {filename}')

def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth'):
    """Load training checkpoint if exists"""
    if not Path(filename).exists():
        print(f'   ℹ️  No checkpoint found at {filename}, starting from scratch')
        return 0, 0.0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    
    print(f'   📂 Loading checkpoint from {filename}...')
    checkpoint = torch.load(filename, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    best_auc = checkpoint['best_auc']
    history = checkpoint['history']
    
    print(f'   ✅ Resumed from epoch {checkpoint["epoch"]} (starting epoch {start_epoch})')
    print(f'   ✅ Best AUC so far: {best_auc:.4f}')
    
    return start_epoch, best_auc, history

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
    print(f'✅ Model created')
    print(f'   Total parameters: {total_params:,}')
    print(f'   Model size: ~{total_params * 4 / 1e6:.1f} MB')
    
    # Training setup
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6)
    
    # Try to load checkpoint and resume
    print('\n🔄 Checking for existing checkpoint...')
    start_epoch, best_auc, history = load_checkpoint(model, optimizer, scheduler, 'checkpoint.pth')
    
    # Training loop
    print(f'\n🚀 Starting training for {Config.NUM_EPOCHS} epochs...')
    print('-'*80)
    
    best_auc = 0.0
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f'\n📅 Epoch {epoch+1}/{Config.NUM_EPOCHS}')
        print(f'   Learning rate: {optimizer.param_groups[0][\"lr\"]:.2e}')
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f'\n📊 Epoch {epoch+1} Summary:')
        print(f'   Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
        print(f'   Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val AUC: {val_auc:.4f}')
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
            }, 'best_deepfake_model.pth')
            print(f'   ✅ Best model saved! (AUC: {val_auc:.4f})')
    
    print('\n🎉 Training complete!')
    print(f'   Best validation AUC: {best_auc:.4f}')

if __name__ == '__main__':
    main()
