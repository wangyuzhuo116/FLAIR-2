"""Dataset implementation for FLAIR-2."""

import json
import numpy as np
import pandas as pd
import rasterio
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FLAIR2Dataset(Dataset):
    """FLAIR-2 dataset for semantic segmentation with dual-stream input."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        aerial_metadata_path: Optional[str] = None,
        centroids_path: Optional[str] = None,
        split_file: Optional[str] = None,
        transform: Optional[A.Compose] = None,
        num_classes: int = 13
    ):
        """Initialize FLAIR-2 dataset.
        
        Args:
            root_dir: Root directory of FLAIR-2 dataset
            split: Dataset split ("train", "test", or "val")
            aerial_metadata_path: Path to aerial metadata JSON
            centroids_path: Path to centroids JSON
            split_file: Optional CSV file with specific samples to use
            transform: Albumentations transform
            num_classes: Number of segmentation classes
        """
        self.root_dir = Path(root_dir)
        self.split = "train" if split == "val" else split  # Val uses train data
        self.transform = transform
        self.num_classes = num_classes
        
        # Load metadata
        self.aerial_metadata = None
        if aerial_metadata_path:
            with open(aerial_metadata_path, 'r') as f:
                self.aerial_metadata = json.load(f)
        
        self.centroids = None
        if centroids_path:
            with open(centroids_path, 'r') as f:
                self.centroids = json.load(f)
        
        # Build sample list
        self.samples = self._build_sample_list(split_file)
    
    def _build_sample_list(self, split_file: Optional[str]) -> List[Dict]:
        """Build list of samples.
        
        Args:
            split_file: Optional CSV file with sample paths
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        if split_file and Path(split_file).exists():
            # Load from split file
            df = pd.read_csv(split_file)
            for _, row in df.iterrows():
                samples.append({
                    'aerial_path': row['aerial_path'],
                    'label_path': row['label_path'],
                    'sen_data_path': row['sen_data_path'],
                    'sen_masks_path': row['sen_masks_path'],
                    'sen_products_path': row['sen_products_path'],
                    'region': row['region'],
                    'zone': row['zone']
                })
        else:
            # Discover from directory structure
            aerial_dir = self.root_dir / self.split / "aerial"
            
            for region_dir in sorted(aerial_dir.glob("D*_*")):
                region_name = region_dir.name
                
                for zone_dir in sorted(region_dir.glob("Z*_*")):
                    zone_name = zone_dir.name
                    img_dir = zone_dir / "img"
                    
                    if not img_dir.exists():
                        continue
                    
                    for img_path in sorted(img_dir.glob("IMG_*.tif")):
                        img_id = img_path.stem.replace("IMG_", "")
                        
                        # Construct paths
                        label_path = self.root_dir / self.split / "labels" / region_name / zone_name / "msk" / f"MSK_{img_id}.tif"
                        sen_dir = self.root_dir / self.split / "sen" / region_name / zone_name / "sen"
                        sen_data_path = sen_dir / f"SEN2_sp_{region_name}-{zone_name}_data.npy"
                        sen_masks_path = sen_dir / f"SEN2_sp_{region_name}-{zone_name}_masks.npy"
                        sen_products_path = sen_dir / f"SEN2_sp_{region_name}-{zone_name}_products.txt"
                        
                        # Check if paths exist
                        if label_path.exists() and sen_data_path.exists():
                            samples.append({
                                'aerial_path': str(img_path),
                                'label_path': str(label_path),
                                'sen_data_path': str(sen_data_path),
                                'sen_masks_path': str(sen_masks_path),
                                'sen_products_path': str(sen_products_path),
                                'region': region_name,
                                'zone': zone_name
                            })
        
        return samples
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)
    
    def _read_aerial(self, path: str) -> np.ndarray:
        """Read aerial image.
        
        Args:
            path: Path to aerial TIF file
            
        Returns:
            Aerial image array (5, H, W) - RGB, NIR, Elevation
        """
        with rasterio.open(path) as src:
            img = src.read()  # (5, H, W)
        return img.astype(np.float32)
    
    def _read_label(self, path: str) -> np.ndarray:
        """Read label mask.
        
        Args:
            path: Path to label TIF file
            
        Returns:
            Label mask array (H, W)
        """
        with rasterio.open(path) as src:
            mask = src.read(1)  # (H, W)
        return mask.astype(np.int64)
    
    def _read_sentinel(self, sample: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Read Sentinel-2 temporal data.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Tuple of (temporal_data, valid_masks)
            - temporal_data: (T, 10, H, W) Sentinel-2 time series
            - valid_masks: (T, H, W) valid pixel masks
        """
        # Load Sentinel-2 data
        sen_data = np.load(sample['sen_data_path'])  # Shape varies
        sen_masks = np.load(sample['sen_masks_path'])  # Shape varies
        
        # Handle different data shapes
        if sen_data.ndim == 3:
            # (10, H, W) - single timestamp
            sen_data = sen_data[np.newaxis, ...]  # (1, 10, H, W)
            sen_masks = sen_masks[np.newaxis, ...]  # (1, H, W)
        elif sen_data.ndim == 4:
            # (T, 10, H, W) - multiple timestamps
            pass
        else:
            raise ValueError(f"Unexpected Sentinel data shape: {sen_data.shape}")
        
        return sen_data.astype(np.float32), sen_masks.astype(np.float32)
    
    def _get_centroid_coords(self, sample: Dict) -> Optional[Tuple[float, float]]:
        """Get patch centroid coordinates.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            (x, y) coordinates or None
        """
        if self.centroids is None:
            return None
        
        # Extract patch ID from path
        img_path = Path(sample['aerial_path'])
        img_id = img_path.stem.replace("IMG_", "")
        key = f"{sample['region']}-{sample['zone']}-{img_id}"
        
        if key in self.centroids:
            return tuple(self.centroids[key])
        
        return None
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with:
                - aerial: (5, H, W) aerial image
                - sentinel: (T, 10, H_s, W_s) sentinel time series
                - sentinel_mask: (T, H_s, W_s) valid pixel masks
                - label: (H, W) segmentation mask
                - metadata: Optional metadata dict
        """
        sample = self.samples[idx]
        
        # Read data
        aerial = self._read_aerial(sample['aerial_path'])  # (5, H, W)
        label = self._read_label(sample['label_path'])  # (H, W)
        sentinel, sentinel_mask = self._read_sentinel(sample)  # (T, 10, H, W), (T, H, W)
        
        # Transpose aerial for albumentations (C, H, W) -> (H, W, C)
        aerial = aerial.transpose(1, 2, 0)  # (H, W, 5)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=aerial, mask=label)
            aerial = transformed['image']  # (5, H, W) after ToTensorV2
            label = transformed['mask']  # (H, W)
        else:
            # Convert to tensor manually
            aerial = torch.from_numpy(aerial.transpose(2, 0, 1))  # (5, H, W)
            label = torch.from_numpy(label)
        
        # Convert sentinel to tensor
        sentinel = torch.from_numpy(sentinel)  # (T, 10, H, W)
        sentinel_mask = torch.from_numpy(sentinel_mask)  # (T, H, W)
        
        result = {
            'aerial': aerial,
            'sentinel': sentinel,
            'sentinel_mask': sentinel_mask,
            'label': label,
        }
        
        # Add metadata if available
        centroid = self._get_centroid_coords(sample)
        if centroid:
            result['centroid'] = torch.tensor(centroid, dtype=torch.float32)
        
        return result


def get_train_transform(config) -> A.Compose:
    """Get training augmentation transform.
    
    Args:
        config: Configuration object
        
    Returns:
        Albumentations composition
    """
    transforms = []
    
    if config.training.augmentation.enabled:
        aug_cfg = config.training.augmentation
        
        if aug_cfg.h_flip > 0:
            transforms.append(A.HorizontalFlip(p=aug_cfg.h_flip))
        
        if aug_cfg.v_flip > 0:
            transforms.append(A.VerticalFlip(p=aug_cfg.v_flip))
        
        if aug_cfg.rotate > 0:
            transforms.append(A.RandomRotate90(p=aug_cfg.rotate))
        
        if aug_cfg.scale_range:
            scale_min, scale_max = aug_cfg.scale_range
            transforms.append(A.RandomScale(scale_limit=(scale_min - 1, scale_max - 1), p=0.5))
        
        if aug_cfg.brightness > 0 or aug_cfg.contrast > 0:
            transforms.append(A.ColorJitter(
                brightness=aug_cfg.brightness,
                contrast=aug_cfg.contrast,
                p=0.5
            ))
    
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def get_val_transform() -> A.Compose:
    """Get validation transform (no augmentation).
    
    Returns:
        Albumentations composition
    """
    return A.Compose([ToTensorV2()])
