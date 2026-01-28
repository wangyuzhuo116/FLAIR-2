"""Data loaders for FLAIR-2 dataset."""

import torch
from torch.utils.data import DataLoader
from typing import Optional

from .dataset import FLAIR2Dataset, get_train_transform, get_val_transform


def create_dataloaders(config):
    """Create train, val, and test dataloaders.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get transforms
    train_transform = get_train_transform(config)
    val_transform = get_val_transform()
    
    # Determine split files
    train_split_file = None
    val_split_file = None
    
    if config.data.use_subset:
        train_split_file = config.data.train_split_file
        val_split_file = config.data.val_split_file
    
    # Create datasets
    train_dataset = FLAIR2Dataset(
        root_dir=config.data.root_dir,
        split="train",
        aerial_metadata_path=config.data.aerial_metadata,
        centroids_path=config.data.centroids_file,
        split_file=train_split_file,
        transform=train_transform,
        num_classes=config.evaluation.num_classes
    )
    
    val_dataset = FLAIR2Dataset(
        root_dir=config.data.root_dir,
        split="val",
        aerial_metadata_path=config.data.aerial_metadata,
        centroids_path=config.data.centroids_file,
        split_file=val_split_file,
        transform=val_transform,
        num_classes=config.evaluation.num_classes
    )
    
    test_dataset = FLAIR2Dataset(
        root_dir=config.data.root_dir,
        split="test",
        aerial_metadata_path=config.data.aerial_metadata,
        centroids_path=config.data.centroids_file,
        split_file=None,  # Use all test data
        transform=val_transform,
        num_classes=config.evaluation.num_classes
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.hardware.num_workers,
        pin_memory=config.hardware.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.hardware.num_workers,
        pin_memory=config.hardware.pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.inference.batch_size,
        shuffle=False,
        num_workers=config.hardware.num_workers,
        pin_memory=config.hardware.pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader
