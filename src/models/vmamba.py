"""VMamba-style Visual State Space Model for aerial imagery.

Self-implemented visual state space backbone inspired by VMamba architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List, Optional


class SS2D(nn.Module):
    """2D Selective Scan module for spatial processing."""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: str = "auto"
    ):
        """Initialize SS2D module.
        
        Args:
            d_model: Model dimension
            d_state: State dimension
            expand: Expansion ratio
            dt_rank: Rank for delta parameter
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        if dt_rank == "auto":
            self.dt_rank = max(16, d_model // 16)
        else:
            self.dt_rank = int(dt_rank)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # State space parameters
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, L, D)
            
        Returns:
            Output tensor (B, L, D)
        """
        B, L, D = x.shape
        
        # Input projection
        x_proj = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = x_proj.chunk(2, dim=-1)  # Each (B, L, d_inner)
        
        # Apply selective scan (simplified version)
        # In full implementation, this would include directional scans
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Simplified state space computation
        y = x + x * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output


class VSSBlock(nn.Module):
    """Visual State Space Block."""
    
    def __init__(
        self,
        hidden_dim: int,
        d_state: int = 16,
        expand: int = 2,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0
    ):
        """Initialize VSS block.
        
        Args:
            hidden_dim: Hidden dimension
            d_state: State space dimension
            expand: Expansion ratio for SS2D
            drop_path: Drop path rate
            mlp_ratio: MLP expansion ratio
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ss2d = SS2D(hidden_dim, d_state=d_state, expand=expand)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, H, W, C)
            
        Returns:
            Output tensor (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        # Flatten spatial dimensions for SS2D
        x_flat = rearrange(x, 'b h w c -> b (h w) c')
        
        # VSS processing
        x = x + self.drop_path(self.ss2d(self.norm1(x_flat)))
        
        # MLP processing
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # Reshape back
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.0):
        """Initialize drop path.
        
        Args:
            drop_prob: Drop probability
        """
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class PatchEmbed(nn.Module):
    """Patch embedding layer."""
    
    def __init__(
        self,
        in_channels: int = 5,
        embed_dim: int = 96,
        patch_size: int = 4
    ):
        """Initialize patch embedding.
        
        Args:
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            patch_size: Patch size
        """
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Embedded tensor (B, H', W', D)
        """
        x = self.proj(x)  # (B, D, H', W')
        x = x.permute(0, 2, 3, 1)  # (B, H', W', D)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """Patch merging layer for downsampling."""
    
    def __init__(self, dim: int):
        """Initialize patch merging.
        
        Args:
            dim: Input dimension
        """
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, H, W, C)
            
        Returns:
            Downsampled tensor (B, H/2, W/2, 2*C)
        """
        B, H, W, C = x.shape
        
        # Pad if necessary
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        # Split into 4 patches and concatenate
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right
        
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, 2*C)
        
        return x


class VMambaStage(nn.Module):
    """VMamba stage with multiple VSS blocks."""
    
    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int = 16,
        expand: int = 2,
        drop_path: List[float] = [],
        mlp_ratio: float = 4.0
    ):
        """Initialize VMamba stage.
        
        Args:
            dim: Feature dimension
            depth: Number of blocks
            d_state: State space dimension
            expand: Expansion ratio
            drop_path: Drop path rates
            mlp_ratio: MLP ratio
        """
        super().__init__()
        
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                d_state=d_state,
                expand=expand,
                drop_path=drop_path[i] if i < len(drop_path) else 0.0,
                mlp_ratio=mlp_ratio
            )
            for i in range(depth)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (B, H, W, C)
            
        Returns:
            Output tensor (B, H, W, C)
        """
        for block in self.blocks:
            x = block(x)
        return x


class VMambaBackbone(nn.Module):
    """VMamba backbone for aerial imagery processing."""
    
    def __init__(
        self,
        in_channels: int = 5,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 9, 2],
        d_state: int = 16,
        ssm_ratio: float = 2.0,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.2,
        patch_size: int = 4
    ):
        """Initialize VMamba backbone.
        
        Args:
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            depths: Number of blocks per stage
            d_state: State space dimension
            ssm_ratio: SSM expansion ratio
            mlp_ratio: MLP expansion ratio
            drop_path_rate: Drop path rate
            patch_size: Patch size for embedding
        """
        super().__init__()
        
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
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
            
            stage = VMambaStage(
                dim=stage_dim,
                depth=stage_depth,
                d_state=d_state,
                expand=int(ssm_ratio),
                drop_path=stage_dpr,
                mlp_ratio=mlp_ratio
            )
            self.stages.append(stage)
            
            # Add merging layer (except for last stage)
            if i < self.num_stages - 1:
                merge = PatchMerging(dim=stage_dim)
                self.merges.append(merge)
        
        # Normalization for each stage output
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim * (2 ** i))
            for i in range(self.num_stages)
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass with multi-scale features.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            List of feature tensors from each stage
        """
        features = []
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, H', W', D)
        
        # Process through stages
        for i in range(self.num_stages):
            x = self.stages[i](x)
            
            # Normalize and save features
            x_norm = self.norms[i](x)
            # Convert to (B, C, H, W) format
            x_feat = x_norm.permute(0, 3, 1, 2)
            features.append(x_feat)
            
            # Merge patches (downsample) if not last stage
            if i < self.num_stages - 1:
                x = self.merges[i](x)
        
        return features
