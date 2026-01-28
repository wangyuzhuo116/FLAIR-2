"""Example usage of FLAIR-2 dual-stream model."""

import torch
from src.utils import load_config
from src.models import build_model

def main():
    """Demonstrate basic usage."""
    print("FLAIR-2 Dual-Stream Model Example")
    print("=" * 80)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config('configs/default.yaml')
    print(f"   Config loaded: {config.logging.exp_name}")
    
    # Build model
    print("\n2. Building model...")
    model = build_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create dummy batch
    print("\n3. Creating dummy batch...")
    batch_size = 2
    aerial_size = 512
    sentinel_size = 40
    
    batch = {
        'aerial': torch.randn(batch_size, 5, aerial_size, aerial_size),
        'sentinel': torch.randn(batch_size, 12, 10, sentinel_size, sentinel_size),
        'sentinel_mask': torch.ones(batch_size, 12, sentinel_size, sentinel_size)
    }
    
    print(f"   Aerial shape: {batch['aerial'].shape}")
    print(f"   Sentinel shape: {batch['sentinel'].shape}")
    
    # Forward pass
    print("\n4. Running forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(batch)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Get predictions
    pred = output.argmax(dim=1)
    print(f"   Prediction shape: {pred.shape}")
    print(f"   Unique classes: {pred.unique().tolist()}")
    
    print("\n" + "=" * 80)
    print("✓ Example completed successfully!")
    print("\nNext steps:")
    print("  1. Prepare your FLAIR-2 dataset")
    print("  2. (Optional) Build 10k subset: python tools/build_subset.py --root-dir <path>")
    print("  3. Train: python -m src.cli train --config configs/default.yaml")
    print("  4. Evaluate: python -m src.cli eval --config configs/default.yaml --checkpoint <path>")


if __name__ == '__main__':
    main()
