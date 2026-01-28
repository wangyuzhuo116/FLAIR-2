"""Fusion module for dual-stream architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between aerial and satellite features."""
    
    def __init__(
        self,
        aerial_dim: int,
        satellite_dim: int,
        num_heads: int = 8,
        out_dim: int = 256
    ):
        """Initialize cross-attention fusion.
        
        Args:
            aerial_dim: Aerial feature dimension
            satellite_dim: Satellite feature dimension
            num_heads: Number of attention heads
            out_dim: Output dimension
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Project features to common dimension
        self.aerial_proj = nn.Linear(aerial_dim, out_dim)
        self.satellite_proj = nn.Linear(satellite_dim, out_dim)
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(out_dim, out_dim)
        self.k_proj = nn.Linear(out_dim, out_dim)
        self.v_proj = nn.Linear(out_dim, out_dim)
        
        # Output projection
        self.out_proj = nn.Linear(out_dim, out_dim)
        
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Linear(out_dim * 4, out_dim)
        )
    
    def forward(
        self,
        aerial_feat: torch.Tensor,
        satellite_feat: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            aerial_feat: Aerial features (B, C_a, H, W)
            satellite_feat: Satellite features (B, C_s, H, W)
            
        Returns:
            Fused features (B, out_dim, H, W)
        """
        B, _, H, W = aerial_feat.shape
        
        # Flatten spatial dimensions
        aerial = rearrange(aerial_feat, 'b c h w -> b (h w) c')
        satellite = rearrange(satellite_feat, 'b c h w -> b (h w) c')
        
        # Project to common dimension
        aerial = self.aerial_proj(aerial)  # (B, HW, D)
        satellite = self.satellite_proj(satellite)  # (B, HW, D)
        
        # Cross-attention: aerial queries, satellite keys/values
        residual = aerial
        
        q = self.q_proj(aerial)  # (B, HW, D)
        k = self.k_proj(satellite)  # (B, HW, D)
        v = self.v_proj(satellite)  # (B, HW, D)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        
        # Residual connection and norm
        out = self.norm1(out + residual)
        
        # FFN
        out = self.norm2(out + self.ffn(out))
        
        # Reshape back to spatial
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        
        return out


class SimpleFusion(nn.Module):
    """Simple concatenation + conv fusion."""
    
    def __init__(
        self,
        aerial_dim: int,
        satellite_dim: int,
        out_dim: int = 256
    ):
        """Initialize simple fusion.
        
        Args:
            aerial_dim: Aerial feature dimension
            satellite_dim: Satellite feature dimension
            out_dim: Output dimension
        """
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(aerial_dim + satellite_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        aerial_feat: torch.Tensor,
        satellite_feat: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            aerial_feat: Aerial features (B, C_a, H, W)
            satellite_feat: Satellite features (B, C_s, H, W)
            
        Returns:
            Fused features (B, out_dim, H, W)
        """
        # Concatenate and fuse
        fused = torch.cat([aerial_feat, satellite_feat], dim=1)
        fused = self.conv(fused)
        return fused
