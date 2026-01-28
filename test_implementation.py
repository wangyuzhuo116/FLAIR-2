"""Quick test script to verify implementation."""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.utils import load_config, Logger, set_seed
        print("  ✓ Utils imported successfully")
    except Exception as e:
        print(f"  ✗ Utils import failed: {e}")
        return False
    
    try:
        from src.data import FLAIR2Dataset, create_dataloaders
        print("  ✓ Data modules imported successfully")
    except Exception as e:
        print(f"  ✗ Data import failed: {e}")
        return False
    
    try:
        from src.models import build_model, VMambaBackbone, STMambaBackbone
        print("  ✓ Model modules imported successfully")
    except Exception as e:
        print(f"  ✗ Model import failed: {e}")
        return False
    
    try:
        from src.training import Trainer, build_loss, MetricTracker
        print("  ✓ Training modules imported successfully")
    except Exception as e:
        print(f"  ✗ Training import failed: {e}")
        return False
    
    return True


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from src.utils import load_config
        config = load_config('configs/default.yaml')
        print(f"  ✓ Config loaded successfully")
        print(f"    - Aerial embed_dim: {config.model.aerial.embed_dim}")
        print(f"    - Satellite embed_dim: {config.model.satellite.embed_dim}")
        print(f"    - Batch size: {config.training.batch_size}")
        return True
    except Exception as e:
        print(f"  ✗ Config loading failed: {e}")
        return False


def test_model_forward():
    """Test model forward pass with dummy data."""
    print("\nTesting model forward pass...")
    
    try:
        from src.utils import load_config
        from src.models import build_model
        
        config = load_config('configs/default.yaml')
        model = build_model(config)
        print(f"  ✓ Model built successfully")
        
        # Create dummy batch
        batch = {
            'aerial': torch.randn(2, 5, 512, 512),
            'sentinel': torch.randn(2, 12, 10, 40, 40),
            'sentinel_mask': torch.ones(2, 12, 40, 40)
        }
        
        # Forward pass
        output = model(batch)
        print(f"  ✓ Forward pass successful")
        print(f"    - Input shape: {batch['aerial'].shape}")
        print(f"    - Output shape: {output.shape}")
        print(f"    - Expected classes: {config.model.decoder.num_classes}")
        
        # Check output shape
        assert output.shape[0] == 2, "Batch dimension mismatch"
        assert output.shape[1] == config.model.decoder.num_classes, "Class dimension mismatch"
        assert output.shape[2] == 512, "Height dimension mismatch"
        assert output.shape[3] == 512, "Width dimension mismatch"
        
        print(f"  ✓ Output shape validation passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_and_metrics():
    """Test loss and metrics computation."""
    print("\nTesting loss and metrics...")
    
    try:
        from src.utils import load_config
        from src.training import build_loss, MetricTracker
        
        config = load_config('configs/default.yaml')
        
        # Build loss
        criterion = build_loss(config)
        print(f"  ✓ Loss function built successfully")
        
        # Build metrics
        metrics = MetricTracker(
            config.evaluation.metrics,
            config.evaluation.num_classes,
            config.training.loss.ignore_index
        )
        print(f"  ✓ Metrics tracker built successfully")
        
        # Test with dummy data
        pred = torch.randn(2, 13, 64, 64)
        target = torch.randint(0, 13, (2, 64, 64))
        
        # Compute loss
        loss = criterion(pred, target)
        print(f"  ✓ Loss computation successful: {loss.item():.4f}")
        
        # Update metrics
        metrics.update(pred, target)
        results = metrics.compute()
        print(f"  ✓ Metrics computation successful:")
        for name, value in results.items():
            if isinstance(value, float):
                print(f"    - {name}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Loss/metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("FLAIR-2 Implementation Tests")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Model Forward Pass", test_model_forward),
        ("Loss and Metrics", test_loss_and_metrics),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Implementation is working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
