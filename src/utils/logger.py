"""Logging utilities for FLAIR-2 project."""

import csv
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Multi-backend logger supporting TensorBoard and CSV."""
    
    def __init__(
        self,
        log_dir: str,
        exp_name: str,
        use_tensorboard: bool = True,
        use_csv: bool = True
    ):
        """Initialize logger.
        
        Args:
            log_dir: Directory for logs
            exp_name: Experiment name
            use_tensorboard: Whether to use TensorBoard
            use_csv: Whether to use CSV logging
        """
        self.log_dir = Path(log_dir) / exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard
        self.use_csv = use_csv
        
        # TensorBoard writer
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(str(self.log_dir / "tensorboard"))
        else:
            self.tb_writer = None
        
        # CSV logging
        if self.use_csv:
            self.csv_path = self.log_dir / "metrics.csv"
            self.csv_file = None
            self.csv_writer = None
            self._init_csv()
        
        self.step = 0
    
    def _init_csv(self):
        """Initialize CSV file."""
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = None  # Will be initialized on first log
        self.csv_fieldnames = None
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log a scalar value.
        
        Args:
            tag: Name of the scalar
            value: Value to log
            step: Step number (uses internal counter if None)
        """
        if step is None:
            step = self.step
        
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)
    
    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ""):
        """Log multiple scalar values.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number (uses internal counter if None)
            prefix: Prefix to add to metric names
        """
        if step is None:
            step = self.step
        
        for name, value in metrics.items():
            tag = f"{prefix}/{name}" if prefix else name
            self.log_scalar(tag, value, step)
        
        # CSV logging
        if self.use_csv and self.csv_writer is not None:
            row = {"step": step}
            for name, value in metrics.items():
                tag = f"{prefix}_{name}" if prefix else name
                row[tag] = value
            
            # Initialize CSV writer if needed
            if self.csv_fieldnames is None:
                self.csv_fieldnames = list(row.keys())
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.csv_fieldnames)
                self.csv_writer.writeheader()
            
            # Write row
            if set(row.keys()) == set(self.csv_fieldnames):
                self.csv_writer.writerow(row)
                self.csv_file.flush()
    
    def log_image(self, tag: str, image: torch.Tensor, step: Optional[int] = None):
        """Log an image.
        
        Args:
            tag: Name of the image
            image: Image tensor (C, H, W) or (B, C, H, W)
            step: Step number (uses internal counter if None)
        """
        if step is None:
            step = self.step
        
        if self.tb_writer is not None:
            self.tb_writer.add_image(tag, image, step)
    
    def log_images(self, tag: str, images: torch.Tensor, step: Optional[int] = None):
        """Log multiple images.
        
        Args:
            tag: Name of the images
            images: Image tensor (B, C, H, W)
            step: Step number (uses internal counter if None)
        """
        if step is None:
            step = self.step
        
        if self.tb_writer is not None:
            self.tb_writer.add_images(tag, images, step)
    
    def increment_step(self):
        """Increment internal step counter."""
        self.step += 1
    
    def close(self):
        """Close logger."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        if self.csv_file is not None:
            self.csv_file.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
