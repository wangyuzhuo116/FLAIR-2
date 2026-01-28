"""Visualization utilities for FLAIR-2 project."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
import cv2


# FLAIR-2 class colors (13 classes)
CLASS_COLORS = np.array([
    [238, 118, 33],   # building
    [245, 245, 82],   # pervious surface
    [255, 0, 0],      # impervious surface
    [194, 143, 61],   # bare soil
    [0, 0, 255],      # water
    [0, 128, 0],      # coniferous
    [144, 238, 144],  # deciduous
    [255, 165, 0],    # brushwood
    [128, 0, 128],    # vineyard
    [173, 255, 47],   # herbaceous vegetation
    [255, 215, 0],    # agricultural land
    [139, 69, 19],    # plowed land
    [128, 128, 128],  # other
])

CLASS_NAMES = [
    "Building",
    "Pervious surface",
    "Impervious surface",
    "Bare soil",
    "Water",
    "Coniferous",
    "Deciduous",
    "Brushwood",
    "Vineyard",
    "Herbaceous vegetation",
    "Agricultural land",
    "Plowed land",
    "Other"
]


def mask_to_rgb(mask: np.ndarray, num_classes: int = 13) -> np.ndarray:
    """Convert semantic mask to RGB image.
    
    Args:
        mask: Semantic mask array (H, W) with class indices
        num_classes: Number of classes
        
    Returns:
        RGB image array (H, W, 3)
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx in range(num_classes):
        rgb[mask == class_idx] = CLASS_COLORS[class_idx]
    
    return rgb


def visualize_prediction(
    aerial_img: np.ndarray,
    gt_mask: Optional[np.ndarray],
    pred_mask: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = False
):
    """Visualize prediction with aerial image, ground truth, and prediction.
    
    Args:
        aerial_img: Aerial image (H, W, 3) RGB uint8
        gt_mask: Ground truth mask (H, W) or None
        pred_mask: Predicted mask (H, W)
        save_path: Path to save visualization
        show: Whether to show plot
    """
    n_cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    if n_cols == 2:
        axes = [axes[0], None, axes[1]]
    
    # Aerial image
    axes[0].imshow(aerial_img)
    axes[0].set_title("Aerial Image")
    axes[0].axis('off')
    
    # Ground truth
    if gt_mask is not None:
        gt_rgb = mask_to_rgb(gt_mask)
        axes[1].imshow(gt_rgb)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
    
    # Prediction
    pred_rgb = mask_to_rgb(pred_mask)
    axes[-1].imshow(pred_rgb)
    axes[-1].set_title("Prediction")
    axes[-1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def create_legend(save_path: str):
    """Create and save class legend.
    
    Args:
        save_path: Path to save legend image
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create color patches
    for i, (color, name) in enumerate(zip(CLASS_COLORS, CLASS_NAMES)):
        ax.barh(i, 1, color=color / 255.0, label=name)
    
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title("FLAIR-2 Class Colors")
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """Overlay colored mask on image.
    
    Args:
        image: RGB image (H, W, 3)
        mask: Semantic mask (H, W)
        alpha: Transparency for overlay
        
    Returns:
        Overlayed image (H, W, 3)
    """
    mask_rgb = mask_to_rgb(mask)
    overlay = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)
    return overlay
