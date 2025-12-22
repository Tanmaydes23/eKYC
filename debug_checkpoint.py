import torch
from pathlib import Path

# Check what's in the checkpoint
checkpoint_path = Path('best_deepfake_model.pth')

if checkpoint_path.exists():
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"File size: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\nCheckpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"\nCheckpoint keys: {checkpoint.keys()}")
        
        if 'model_state_dict' in checkpoint:
            print("\n✅ Found 'model_state_dict' key")
            state_dict = checkpoint['model_state_dict']
        else:
            print("\n⚠️ No 'model_state_dict' key, using checkpoint directly")
            state_dict = checkpoint
        
        print(f"\nState dict keys (first 10):")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            print(f"  {i+1}. {key}: {state_dict[key].shape if hasattr(state_dict[key], 'shape') else type(state_dict[key])}")
        
        print(f"\nTotal parameters in checkpoint: {len(state_dict)}")
    else:
        print(f"\n⚠️ Checkpoint is not a dict: {type(checkpoint)}")

else:
    print(f"❌ Checkpoint file not found: {checkpoint_path}")
