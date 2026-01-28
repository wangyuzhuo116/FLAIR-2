"""Trainer for FLAIR-2 model."""

import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from ..models import build_model
from ..utils import Logger, set_seed, save_config
from .losses import build_loss
from .metrics import MetricTracker


class Trainer:
    """Trainer class for FLAIR-2 model."""
    
    def __init__(
        self,
        config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: Logger,
        device: str = "cuda"
    ):
        """Initialize trainer.
        
        Args:
            config: Configuration object
            train_loader: Training data loader
            val_loader: Validation data loader
            logger: Logger instance
            device: Device to use
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.device = device
        
        # Set seed
        set_seed(config.hardware.seed)
        
        # Build model
        self.model = build_model(config).to(device)
        
        # Build loss
        self.criterion = build_loss(config).to(device)
        
        # Build optimizer
        self.optimizer = self._build_optimizer()
        
        # Build scheduler
        self.scheduler = self._build_scheduler()
        
        # Mixed precision training
        self.use_amp = config.training.use_amp
        self.scaler = GradScaler() if self.use_amp else None
        
        # Metrics
        self.train_metrics = MetricTracker(
            config.evaluation.metrics,
            config.evaluation.num_classes,
            config.training.loss.ignore_index
        )
        self.val_metrics = MetricTracker(
            config.evaluation.metrics,
            config.evaluation.num_classes,
            config.training.loss.ignore_index
        )
        
        # Checkpoint
        self.checkpoint_dir = Path(config.checkpoint.save_dir) / config.logging.exp_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_metric = 0.0
        self.current_epoch = 0
    
    def _build_optimizer(self):
        """Build optimizer."""
        optimizer_type = self.config.training.optimizer
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay
        
        if optimizer_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        scheduler_type = self.config.training.scheduler
        num_epochs = self.config.training.num_epochs
        warmup_epochs = self.config.training.warmup_epochs
        
        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs - warmup_epochs
            )
        elif scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_type is None:
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    def train_epoch(self) -> dict:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            target = batch['label']
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    output = self.model(batch)
                    loss = self.criterion(output, target)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(batch)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.train_metrics.update(output.detach(), target)
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log to tensorboard
            if batch_idx % self.config.logging.log_interval == 0:
                global_step = self.current_epoch * num_batches + batch_idx
                self.logger.log_scalar("train/loss_step", loss.item(), global_step)
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        metrics = self.train_metrics.compute()
        metrics['loss'] = avg_loss
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        for batch in pbar:
            # Move data to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            target = batch['label']
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    output = self.model(batch)
                    loss = self.criterion(output, target)
            else:
                output = self.model(batch)
                loss = self.criterion(output, target)
            
            # Update metrics
            total_loss += loss.item()
            self.val_metrics.update(output, target)
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Compute epoch metrics
        avg_loss = total_loss / num_batches
        metrics = self.val_metrics.compute()
        metrics['loss'] = avg_loss
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.to_dict()
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save last checkpoint
        last_path = self.checkpoint_dir / "last.pth"
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
        
        # Save epoch checkpoint
        if self.current_epoch % 10 == 0:
            epoch_path = self.checkpoint_dir / f"epoch_{self.current_epoch}.pth"
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.training.num_epochs} epochs")
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            self.logger.log_scalars(train_metrics, epoch, prefix="train")
            self.logger.log_scalars(val_metrics, epoch, prefix="val")
            
            # Print metrics
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, mIoU: {train_metrics.get('miou', 0):.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, mIoU: {val_metrics.get('miou', 0):.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Save checkpoint
            monitor_metric = val_metrics.get(
                self.config.checkpoint.monitor.replace('val_', ''),
                val_metrics['loss']
            )
            
            is_best = False
            if self.config.checkpoint.mode == "max":
                is_best = monitor_metric > self.best_metric
            else:
                is_best = monitor_metric < self.best_metric
            
            if is_best:
                self.best_metric = monitor_metric
            
            self.save_checkpoint(is_best=is_best)
        
        print(f"\nTraining completed!")
        print(f"Best {self.config.checkpoint.monitor}: {self.best_metric:.4f}")
