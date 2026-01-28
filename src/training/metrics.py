"""Metrics for semantic segmentation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class IoU:
    """Intersection over Union metric."""
    
    def __init__(self, num_classes: int, ignore_index: int = 255):
        """Initialize IoU metric.
        
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore in computation
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metric with predictions and targets.
        
        Args:
            pred: Predicted class indices (B, H, W)
            target: Target class indices (B, H, W)
        """
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        # Filter out ignore index
        valid_mask = target != self.ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # Compute intersection and union for each class
        for cls in range(self.num_classes):
            pred_mask = pred == cls
            target_mask = target == cls
            
            self.intersection[cls] += np.logical_and(pred_mask, target_mask).sum()
            self.union[cls] += np.logical_or(pred_mask, target_mask).sum()
    
    def compute(self) -> dict:
        """Compute IoU metrics.
        
        Returns:
            Dictionary with mIoU and per-class IoU
        """
        iou = self.intersection / (self.union + 1e-10)
        
        # Compute mean IoU (only over classes that appear)
        valid_classes = self.union > 0
        miou = iou[valid_classes].mean()
        
        return {
            'miou': miou,
            'iou_per_class': iou
        }


class Accuracy:
    """Pixel accuracy metric."""
    
    def __init__(self, ignore_index: int = 255):
        """Initialize accuracy metric.
        
        Args:
            ignore_index: Index to ignore in computation
        """
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.correct = 0
        self.total = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metric.
        
        Args:
            pred: Predicted class indices (B, H, W)
            target: Target class indices (B, H, W)
        """
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        # Filter out ignore index
        valid_mask = target != self.ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        self.correct += (pred == target).sum()
        self.total += len(pred)
    
    def compute(self) -> dict:
        """Compute accuracy.
        
        Returns:
            Dictionary with accuracy
        """
        acc = self.correct / (self.total + 1e-10)
        return {'accuracy': acc}


class F1Score:
    """F1 score metric."""
    
    def __init__(self, num_classes: int, ignore_index: int = 255):
        """Initialize F1 score metric.
        
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore in computation
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.tp = np.zeros(self.num_classes)
        self.fp = np.zeros(self.num_classes)
        self.fn = np.zeros(self.num_classes)
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metric.
        
        Args:
            pred: Predicted class indices (B, H, W)
            target: Target class indices (B, H, W)
        """
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        # Filter out ignore index
        valid_mask = target != self.ignore_index
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        # Compute TP, FP, FN for each class
        for cls in range(self.num_classes):
            pred_mask = pred == cls
            target_mask = target == cls
            
            self.tp[cls] += np.logical_and(pred_mask, target_mask).sum()
            self.fp[cls] += np.logical_and(pred_mask, ~target_mask).sum()
            self.fn[cls] += np.logical_and(~pred_mask, target_mask).sum()
    
    def compute(self) -> dict:
        """Compute F1 score.
        
        Returns:
            Dictionary with mean F1 and per-class F1
        """
        precision = self.tp / (self.tp + self.fp + 1e-10)
        recall = self.tp / (self.tp + self.fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        # Compute mean F1 (only over classes that appear)
        valid_classes = (self.tp + self.fn) > 0
        mean_f1 = f1[valid_classes].mean()
        
        return {
            'f1': mean_f1,
            'f1_per_class': f1
        }


class MetricTracker:
    """Track multiple metrics."""
    
    def __init__(self, metrics: list, num_classes: int, ignore_index: int = 255):
        """Initialize metric tracker.
        
        Args:
            metrics: List of metric names
            num_classes: Number of classes
            ignore_index: Index to ignore
        """
        self.metrics = {}
        
        for metric_name in metrics:
            if metric_name == 'miou':
                self.metrics[metric_name] = IoU(num_classes, ignore_index)
            elif metric_name == 'accuracy':
                self.metrics[metric_name] = Accuracy(ignore_index)
            elif metric_name == 'f1':
                self.metrics[metric_name] = F1Score(num_classes, ignore_index)
            else:
                raise ValueError(f"Unknown metric: {metric_name}")
    
    def reset(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update all metrics.
        
        Args:
            pred: Predicted logits (B, C, H, W) or indices (B, H, W)
            target: Target class indices (B, H, W)
        """
        # Convert logits to predictions if needed
        if pred.ndim == 4:
            pred = pred.argmax(dim=1)
        
        # Update each metric
        for metric in self.metrics.values():
            metric.update(pred, target)
    
    def compute(self) -> dict:
        """Compute all metrics.
        
        Returns:
            Dictionary of metric values
        """
        results = {}
        for name, metric in self.metrics.items():
            results.update(metric.compute())
        return results
