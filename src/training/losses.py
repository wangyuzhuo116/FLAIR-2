"""Loss functions for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss with optional class weights."""
    
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        reduction: str = 'mean'
    ):
        """Initialize cross-entropy loss.
        
        Args:
            weight: Class weights
            ignore_index: Index to ignore
            reduction: Reduction method
        """
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Target labels (B, H, W)
            
        Returns:
            Loss value
        """
        return F.cross_entropy(
            pred,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        smooth: float = 1.0
    ):
        """Initialize dice loss.
        
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore
            smooth: Smoothing factor
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Target labels (B, H, W)
            
        Returns:
            Loss value
        """
        # Get probabilities
        pred = F.softmax(pred, dim=1)
        
        # One-hot encode target
        target_one_hot = F.one_hot(
            target.clamp(0, self.num_classes - 1),
            num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()
        
        # Mask out ignore index
        valid_mask = (target != self.ignore_index).unsqueeze(1).float()
        pred = pred * valid_mask
        target_one_hot = target_one_hot * valid_mask
        
        # Compute dice
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice.mean()
        
        return loss


class CombinedLoss(nn.Module):
    """Combined loss (CE + Dice)."""
    
    def __init__(
        self,
        num_classes: int,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0
    ):
        """Initialize combined loss.
        
        Args:
            num_classes: Number of classes
            weight: Class weights for CE
            ignore_index: Index to ignore
            ce_weight: Weight for CE loss
            dice_weight: Weight for Dice loss
        """
        super().__init__()
        self.ce_loss = CrossEntropyLoss(weight, ignore_index)
        self.dice_loss = DiceLoss(num_classes, ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Target labels (B, H, W)
            
        Returns:
            Loss value
        """
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        return self.ce_weight * ce + self.dice_weight * dice


def build_loss(config, class_weights: Optional[torch.Tensor] = None):
    """Build loss function from config.
    
    Args:
        config: Configuration object
        class_weights: Optional class weights
        
    Returns:
        Loss function
    """
    loss_type = config.training.loss.type
    num_classes = config.evaluation.num_classes
    ignore_index = config.training.loss.ignore_index
    
    # Use config weights if provided, otherwise use computed weights
    if config.training.loss.class_weights is not None:
        weight = torch.tensor(config.training.loss.class_weights)
    else:
        weight = class_weights
    
    if loss_type == "cross_entropy":
        return CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif loss_type == "dice":
        return DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
    elif loss_type == "combined":
        return CombinedLoss(
            num_classes=num_classes,
            weight=weight,
            ignore_index=ignore_index
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
