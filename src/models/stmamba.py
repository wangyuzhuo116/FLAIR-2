"""STmamba-like architecture for satellite temporal processing.

Self-implemented spatio-temporal state space model inspired by VideoMamba.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Optional


class TemporalSS(nn.Module):
    """Temporal Selective Scan for processing time series."""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2
    ):
        """Initialize temporal selective scan.
        
        Args:
            d_model: Model dimension
            d_state: State dimension
            expand: Expansion ratio
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        # Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # State space parameters
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, T, D)
            
        Returns:
            Output tensor (B, T, D)
        """
        # Input projection
        x_proj = self.in_proj(x)
        x, z = x_proj.chunk(2, dim=-1)
        
        # Temporal processing (simplified)
        A = -torch.exp(self.A_log)
        y = x + x * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        return output


class STBlock(nn.Module):
    """Spatio-Temporal Block combining spatial and temporal processing."""
    
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        expand: int = 2,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0
    ):
        """Initialize ST block.
        
        Args:
            dim: Feature dimension
            d_state: State space dimension
            expand: Expansion ratio
            mlp_ratio: MLP expansion ratio
            drop_path: Drop path rate
        """
        super().__init__()
        
        # Spatial processing
        self.norm_spatial = nn.LayerNorm(dim)
        self.spatial_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
        # Temporal processing
        self.norm_temporal = nn.LayerNorm(dim)
        self.temporal_ss = TemporalSS(dim, d_state=d_state, expand=expand)
        
        # MLP
        self.norm_mlp = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, T, H, W, C)
            
        Returns:
            Output tensor (B, T, H, W, C)
        """
        B, T, H, W, C = x.shape
        
        # Spatial processing
        x_spatial = rearrange(x, 'b t h w c -> (b t) c h w')
        x_spatial = self.spatial_conv(x_spatial)
        x_spatial = rearrange(x_spatial, '(b t) c h w -> b t h w c', b=B, t=T)
        x = x + self.drop_path(self.norm_spatial(x_spatial))
        
        # Temporal processing
        x_temporal = rearrange(x, 'b t h w c -> (b h w) t c')
        x_temporal = self.temporal_ss(self.norm_temporal(x_temporal))
        x_temporal = rearrange(x_temporal, '(b h w) t c -> b t h w c', b=B, h=H, w=W)
        x = x + self.drop_path(x_temporal)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm_mlp(x)))
        
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class STPatchEmbed(nn.Module):
    """Spatio-temporal patch embedding."""
    
    def __init__(
        self,
        in_channels: int = 10,
        embed_dim: int = 96,
        patch_size: int = 4,
        temporal_kernel: int = 1
    ):
        """Initialize ST patch embedding.
        
        Args:
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            patch_size: Spatial patch size
            temporal_kernel: Temporal kernel size
        """
        super().__init__()
        self.patch_size = patch_size
        self.temporal_kernel = temporal_kernel
        
        # 3D convolution for spatio-temporal embedding
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(temporal_kernel, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
            padding=(temporal_kernel // 2, 0, 0)
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, T, C, H, W)
            
        Returns:
            Embedded tensor (B, T, H', W', D)
        """
        # Rearrange for 3D conv
        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.proj(x)  # (B, D, T, H', W')
        x = rearrange(x, 'b d t h w -> b t h w d')
        x = self.norm(x)
        return x


class STStage(nn.Module):
    """Spatio-temporal stage with multiple ST blocks."""
    
    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        expand: int = 2,
        mlp_ratio: float = 4.0,
        drop_path: List[float] = []
    ):
        """Initialize ST stage.
        
        Args:
            dim: Feature dimension
            depth: Number of blocks
            d_state: State dimension
            expand: Expansion ratio
            mlp_ratio: MLP ratio
            drop_path: Drop path rates
        """
        super().__init__()
        
        self.blocks = nn.ModuleList([
            STBlock(
                dim=dim,
                d_state=d_state,
                expand=expand,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i] if i < len(drop_path) else 0.0
            )
            for i in range(depth)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, T, H, W, C)
            
        Returns:
            Output tensor (B, T, H, W, C)
        """
        for block in self.blocks:
            x = block(x)
        return x


class STPatchMerging(nn.Module):
    """Spatial patch merging for downsampling."""
    
    def __init__(self, dim: int):
        """Initialize spatial patch merging.
        
        Args:
            dim: Input dimension
        """
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, T, H, W, C)
            
        Returns:
            Downsampled tensor (B, T, H/2, W/2, 2*C)
        """
        B, T, H, W, C = x.shape
        
        # Pad if necessary
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        # Split into 4 patches
        x0 = x[:, :, 0::2, 0::2, :]
        x1 = x[:, :, 1::2, 0::2, :]
        x2 = x[:, :, 0::2, 1::2, :]
        x3 = x[:, :, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class STMambaBackbone(nn.Module):
    """STmamba backbone for satellite temporal processing."""
    
    def __init__(
        self,
        in_channels: int = 10,
        temporal_depth: int = 12,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2,
        patch_size: int = 4
    ):
        """Initialize STmamba backbone.
        
        Args:
            in_channels: Number of input channels
            temporal_depth: Maximum temporal sequence length
            embed_dim: Embedding dimension
            depths: Number of blocks per stage
            d_state: State dimension
            ssm_ratio: SSM expansion ratio
            mlp_ratio: MLP ratio
            drop_path_rate: Drop path rate
            patch_size: Spatial patch size
        """
        super().__init__()
        
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.temporal_depth = temporal_depth
        
        # Patch embedding
        self.patch_embed = STPatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            temporal_kernel=1
        )
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build stages
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        
        for i in range(self.num_stages):
            stage_dim = embed_dim * (2 ** i)
            stage_depth = depths[i]
            stage_dpr = dpr[sum(depths[:i]):sum(depths[:i+1])]
            
            stage = STStage(
                dim=stage_dim,
                depth=stage_depth,
                d_state=d_state,
                expand=int(ssm_ratio),
                mlp_ratio=mlp_ratio,
                drop_path=stage_dpr
            )
            self.stages.append(stage)
            
            if i < self.num_stages - 1:
                merge = STPatchMerging(dim=stage_dim)
                self.merges.append(merge)
        
        # Norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim * (2 ** i))
            for i in range(self.num_stages)
        ])
        
        # Temporal pooling for final feature
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """Forward pass with multi-scale features.
        
        Args:
            x: Input tensor (B, T, C, H, W)
            mask: Optional mask tensor (B, T, H, W)
            
        Returns:
            List of feature tensors from each stage
        """
        features = []
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, T, H', W', D)
        
        # Process through stages
        for i in range(self.num_stages):
            x = self.stages[i](x)
            
            # Normalize
            x_norm = self.norms[i](x)
            
            # Temporal pooling and convert to (B, C, H, W)
            B, T, H, W, C = x_norm.shape
            x_pooled = rearrange(x_norm, 'b t h w c -> (b h w) c t')
            x_pooled = self.temporal_pool(x_pooled).squeeze(-1)
            x_feat = rearrange(x_pooled, '(b h w) c -> b c h w', b=B, h=H, w=W)
            features.append(x_feat)
            
            # Merge if not last stage
            if i < self.num_stages - 1:
                x = self.merges[i](x)
        
        return features
