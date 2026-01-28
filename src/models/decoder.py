"""Decoder module for segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class UpsampleBlock(nn.Module):
    """Upsampling block with skip connections."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int
    ):
        """Initialize upsample block.
        
        Args:
            in_channels: Input channels
            skip_channels: Skip connection channels
            out_channels: Output channels
        """
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=2,
            stride=2
        )
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features (B, C_in, H, W)
            skip: Skip connection features (B, C_skip, 2H, 2W)
            
        Returns:
            Output features (B, C_out, 2H, 2W)
        """
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class SegmentationDecoder(nn.Module):
    """Decoder for semantic segmentation."""
    
    def __init__(
        self,
        in_channels: int = 256,
        skip_channels: List[int] = [192, 192, 96, 96],
        decoder_channels: List[int] = [256, 128, 64, 32],
        num_classes: int = 13
    ):
        """Initialize segmentation decoder.
        
        Args:
            in_channels: Input channels from fusion
            skip_channels: Channels from skip connections (from stages)
            decoder_channels: Decoder channel progression
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.num_stages = len(decoder_channels)
        
        # Build upsample blocks
        self.blocks = nn.ModuleList()
        
        for i in range(self.num_stages):
            in_ch = in_channels if i == 0 else decoder_channels[i-1]
            skip_ch = skip_channels[i] if i < len(skip_channels) else 0
            out_ch = decoder_channels[i]
            
            self.blocks.append(UpsampleBlock(in_ch, skip_ch, out_ch))
        
        # Final classifier
        self.final_upsample = nn.ConvTranspose2d(
            decoder_channels[-1],
            decoder_channels[-1],
            kernel_size=4,
            stride=4
        )
        
        self.classifier = nn.Conv2d(
            decoder_channels[-1],
            num_classes,
            kernel_size=1
        )
    
    def forward(
        self,
        x: torch.Tensor,
        skip_connections: List[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Fused features (B, C, H, W)
            skip_connections: List of skip features from encoder stages
            
        Returns:
            Segmentation logits (B, num_classes, H_orig, W_orig)
        """
        # Reverse skip connections (from deep to shallow)
        skip_connections = skip_connections[::-1]
        
        # Progressive upsampling with skip connections
        for i, block in enumerate(self.blocks):
            if i < len(skip_connections):
                skip = skip_connections[i]
            else:
                # No skip connection, use zeros
                skip = torch.zeros(
                    x.shape[0],
                    0,
                    x.shape[2] * 2,
                    x.shape[3] * 2,
                    device=x.device
                )
            x = block(x, skip)
        
        # Final upsampling to match input resolution
        x = self.final_upsample(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
