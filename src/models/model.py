"""Complete dual-stream model for FLAIR-2."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .vmamba import VMambaBackbone
from .stmamba import STMambaBackbone
from .fusion import CrossAttentionFusion, SimpleFusion
from .decoder import SegmentationDecoder


class DualStreamModel(nn.Module):
    """Dual-stream model with VMamba and STmamba backbones."""
    
    def __init__(self, config):
        """Initialize dual-stream model.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        # Aerial stream (VMamba)
        self.aerial_backbone = VMambaBackbone(
            in_channels=config.model.aerial.in_channels,
            embed_dim=config.model.aerial.embed_dim,
            depths=config.model.aerial.depths,
            d_state=config.model.aerial.ssm_d_state,
            ssm_ratio=config.model.aerial.ssm_ratio,
            mlp_ratio=config.model.aerial.mlp_ratio,
            drop_path_rate=config.model.aerial.drop_path_rate,
            patch_size=4
        )
        
        # Satellite stream (STmamba)
        self.satellite_backbone = STMambaBackbone(
            in_channels=config.model.satellite.in_channels,
            temporal_depth=config.model.satellite.temporal_depth,
            embed_dim=config.model.satellite.embed_dim,
            depths=config.model.satellite.depths,
            d_state=config.model.satellite.ssm_d_state,
            ssm_ratio=config.model.satellite.ssm_ratio,
            mlp_ratio=config.model.satellite.mlp_ratio,
            drop_path_rate=config.model.satellite.drop_path_rate,
            patch_size=4
        )
        
        # Get feature dimensions from last stage
        aerial_dim = config.model.aerial.embed_dim * (2 ** (len(config.model.aerial.depths) - 1))
        satellite_dim = config.model.satellite.embed_dim * (2 ** (len(config.model.satellite.depths) - 1))
        
        # Fusion module
        fusion_type = config.model.fusion.type
        fusion_dim = config.model.fusion.dim
        
        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                aerial_dim=aerial_dim,
                satellite_dim=satellite_dim,
                num_heads=config.model.fusion.num_heads,
                out_dim=fusion_dim
            )
        else:
            self.fusion = SimpleFusion(
                aerial_dim=aerial_dim,
                satellite_dim=satellite_dim,
                out_dim=fusion_dim
            )
        
        # Prepare skip connection channels
        # Aerial and satellite features are concatenated at each stage
        skip_channels = []
        for i in range(len(config.model.aerial.depths) - 1):
            aerial_ch = config.model.aerial.embed_dim * (2 ** i)
            satellite_ch = config.model.satellite.embed_dim * (2 ** i)
            skip_channels.append(aerial_ch + satellite_ch)
        
        # Decoder
        self.decoder = SegmentationDecoder(
            in_channels=fusion_dim,
            skip_channels=skip_channels[::-1],  # Reverse for decoder
            decoder_channels=config.model.decoder.channels,
            num_classes=config.model.decoder.num_classes
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.
        
        Args:
            batch: Dictionary containing:
                - aerial: (B, 5, H, W)
                - sentinel: (B, T, 10, H_s, W_s)
                - sentinel_mask: (B, T, H_s, W_s)
                
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        aerial = batch['aerial']
        sentinel = batch['sentinel']
        sentinel_mask = batch.get('sentinel_mask', None)
        
        # Extract multi-scale features from both streams
        aerial_features = self.aerial_backbone(aerial)  # List of features
        satellite_features = self.satellite_backbone(sentinel, sentinel_mask)  # List of features
        
        # Resize satellite features to match aerial features at each stage
        aligned_satellite_features = []
        for sat_feat, aer_feat in zip(satellite_features, aerial_features):
            if sat_feat.shape[2:] != aer_feat.shape[2:]:
                sat_feat = F.interpolate(
                    sat_feat,
                    size=aer_feat.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )
            aligned_satellite_features.append(sat_feat)
        
        # Fuse last-stage features
        fused_features = self.fusion(
            aerial_features[-1],
            aligned_satellite_features[-1]
        )
        
        # Prepare skip connections (concatenate aerial + satellite at each stage)
        skip_connections = []
        for i in range(len(aerial_features) - 1):
            skip = torch.cat([aerial_features[i], aligned_satellite_features[i]], dim=1)
            skip_connections.append(skip)
        
        # Decode
        logits = self.decoder(fused_features, skip_connections)
        
        # Resize to match input resolution if needed
        if logits.shape[2:] != aerial.shape[2:]:
            logits = F.interpolate(
                logits,
                size=aerial.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        
        return logits


def build_model(config):
    """Build model from config.
    
    Args:
        config: Configuration object
        
    Returns:
        DualStreamModel instance
    """
    return DualStreamModel(config)
