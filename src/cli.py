"""Command-line interface for FLAIR-2 project."""

import argparse
import torch
import sys
from pathlib import Path

from .utils import load_config, Logger
from .data import create_dataloaders
from .training import Trainer
from .models import build_model


def train_command(args):
    """Train model."""
    # Load config
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    
    # Create logger
    logger = Logger(
        log_dir=config.logging.log_dir,
        exp_name=config.logging.exp_name,
        use_tensorboard=config.logging.tensorboard,
        use_csv=config.logging.csv_log
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        device=device
    )
    
    # Load checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    # Close logger
    logger.close()


def eval_command(args):
    """Evaluate model."""
    # Load config
    config = load_config(args.config)
    
    # Create dataloaders
    print("Creating dataloaders...")
    _, val_loader, test_loader = create_dataloaders(config)
    
    # Use test loader if specified
    eval_loader = test_loader if args.test else val_loader
    print(f"Eval samples: {len(eval_loader.dataset)}")
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create metrics
    from .training import MetricTracker
    metrics = MetricTracker(
        config.evaluation.metrics,
        config.evaluation.num_classes,
        config.training.loss.ignore_index
    )
    
    # Evaluate
    model.eval()
    metrics.reset()
    
    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward
            output = model(batch)
            
            # Update metrics
            metrics.update(output, batch['label'])
    
    # Compute and print metrics
    results = metrics.compute()
    print("\nEvaluation Results:")
    for name, value in results.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")


def inference_command(args):
    """Run inference."""
    # Load config
    config = load_config(args.config)
    
    # Override output dir
    if args.output_dir:
        config.inference.output_dir = args.output_dir
    
    # Create dataloader
    print("Creating dataloader...")
    _, _, test_loader = create_dataloaders(config)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Run inference
    from .utils import visualize_prediction
    import numpy as np
    import rasterio
    from tqdm import tqdm
    
    output_dir = Path(config.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
            # Move to device
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            
            # Forward
            output = model(batch_gpu)
            pred = output.argmax(dim=1).cpu().numpy()
            
            # Save predictions
            for i in range(pred.shape[0]):
                sample_id = batch_idx * config.inference.batch_size + i
                pred_path = output_dir / f"pred_{sample_id:06d}.tif"
                
                # Save as TIF
                with rasterio.open(
                    pred_path,
                    'w',
                    driver='GTiff',
                    height=pred.shape[1],
                    width=pred.shape[2],
                    count=1,
                    dtype=pred.dtype
                ) as dst:
                    dst.write(pred[i], 1)
    
    print(f"\nPredictions saved to: {output_dir}")


def visualize_command(args):
    """Visualize predictions."""
    # Load config
    config = load_config(args.config)
    
    # Create dataloader
    print("Creating dataloader...")
    _, val_loader, _ = create_dataloaders(config)
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Visualize
    from .utils import visualize_prediction
    import numpy as np
    
    vis_dir = Path(config.visualization.save_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    
    num_visualized = 0
    max_vis = config.visualization.num_samples
    
    with torch.no_grad():
        for batch in val_loader:
            if num_visualized >= max_vis:
                break
            
            # Move to device
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
            
            # Forward
            output = model(batch_gpu)
            pred = output.argmax(dim=1).cpu().numpy()
            
            # Get data
            aerial = batch['aerial'].cpu().numpy()
            label = batch['label'].cpu().numpy()
            
            # Visualize each sample in batch
            for i in range(aerial.shape[0]):
                if num_visualized >= max_vis:
                    break
                
                # Convert aerial to RGB (first 3 channels)
                aerial_rgb = aerial[i, :3].transpose(1, 2, 0)
                aerial_rgb = (aerial_rgb * 255).clip(0, 255).astype(np.uint8)
                
                # Visualize
                save_path = vis_dir / f"vis_{num_visualized:04d}.png"
                visualize_prediction(
                    aerial_rgb,
                    label[i],
                    pred[i],
                    save_path=str(save_path)
                )
                
                num_visualized += 1
            
            if num_visualized >= max_vis:
                break
    
    print(f"\nVisualizations saved to: {vis_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="FLAIR-2 Dual-Stream Segmentation")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--config', type=str, required=True, help='Config file path')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    train_parser.add_argument('--batch-size', type=int, help='Override batch size')
    train_parser.add_argument('--epochs', type=int, help='Override number of epochs')
    train_parser.add_argument('--lr', type=float, help='Override learning rate')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate model')
    eval_parser.add_argument('--config', type=str, required=True, help='Config file path')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    eval_parser.add_argument('--test', action='store_true', help='Use test set instead of val')
    
    # Inference command
    infer_parser = subparsers.add_parser('inference', help='Run inference')
    infer_parser.add_argument('--config', type=str, required=True, help='Config file path')
    infer_parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    infer_parser.add_argument('--output-dir', type=str, help='Output directory')
    
    # Visualize command
    vis_parser = subparsers.add_parser('visualize', help='Visualize predictions')
    vis_parser.add_argument('--config', type=str, required=True, help='Config file path')
    vis_parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    
    # Parse args
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'eval':
        eval_command(args)
    elif args.command == 'inference':
        inference_command(args)
    elif args.command == 'visualize':
        visualize_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
