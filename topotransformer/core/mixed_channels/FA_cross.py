"""
FA_cross.py - Cross-attention variant of FA.py

Architecture:
- First block in each stage uses cross-attention:
  - Q: from ALL channels (projected to dim)
  - K, V: from conditional channels only (projected to dim)
- Remaining blocks use standard self-attention on all channels
- Conditional channels are only used at input level for the first cross-attention block
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from natten.functional import na2d
from timm.models.layers import DropPath

from pdetransformer.core.mixed_channels.udit import FinalLayer
from .FA import (
    AdaLayerNormZero,
    LabelEmbedder,
    OverlapPatchEmbed,
    SimplePatchEmbed,
    TimestepEmbedder,
    Downsample,
    Upsample,
    Mlp,
    WindowAttention2DTime_NATTEN,
)


class CrossNeighborhoodAttention2D(nn.Module):
    """
    NATTEN cross-attention for first block:
    - Q comes from ALL channels (projected to dim)
    - K/V come from conditional channels (projected to dim)
    
    All projections go to the same embedding dimension to satisfy NATTEN's
    requirement that head_dim be divisible by 8.
    """

    def __init__(
        self,
        dim: int,  # full embedding dimension (e.g., 96)
        dim_kv: int,  # number of conditional channels (e.g., 1)
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        resolution: int = 8,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dim_kv = dim_kv
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.preferred_kernel_size = resolution
        self.stride = stride

        # Q projection: dim -> dim (identity-like, but trainable)
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=qkv_bias)
        # K/V projection: dim_kv -> dim (project conditional channels to full dim)
        self.k_proj = nn.Linear(dim_kv, num_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim_kv, num_heads * self.head_dim, bias=qkv_bias)
        
        self.proj = nn.Linear(num_heads * self.head_dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_kernel_stride(self, H: int, W: int):
        kernel = min(self.preferred_kernel_size, H, W)
        if kernel < 2:
            kernel = 2
        elif kernel % 2 == 0 and kernel > 2:
            kernel -= 1
        kernel = min(kernel, H, W)
        return min(self.stride, kernel), kernel

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, dim) all channels (query source)
            x_cond: (B, H, W, dim_kv) conditional channels (key/value source)
        Returns:
            (B, H, W, dim) cross-attended output
        """
        B, H, W, _ = x.shape
        stride, kernel = self._get_kernel_stride(H, W)

        q = self.q_proj(x).view(B, H, W, self.num_heads, self.head_dim)
        k = self.k_proj(x_cond).view(B, H, W, self.num_heads, self.head_dim)
        v = self.v_proj(x_cond).view(B, H, W, self.num_heads, self.head_dim)

        # NATTEN functional API expects (B, H, W, heads, head_dim)
        out = na2d(q, k, v, kernel_size=kernel, dilation=1, stride=stride)
        out = out.reshape(B, H, W, self.num_heads * self.head_dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class PDEBlock_CrossAttn(nn.Module):
    """
    First block with cross-attention:
    - Q from all channels
    - K/V from conditional channels only
    """

    def __init__(
        self,
        dim: int,
        conditional_channels: Sequence[int],
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        window_size: int = 7,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.conditional_channels = sorted(list(conditional_channels))
        self.window_size = window_size
        self.stride = stride

        dim_kv = len(self.conditional_channels)
        assert dim_kv > 0, "There must be at least one conditional channel for keys/values."

        self.norm1 = norm_layer(dim)
        self.attn = CrossNeighborhoodAttention2D(
            dim=dim,
            dim_kv=dim_kv,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            resolution=window_size,
            stride=stride,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # DiT conditioning
        self.adain = AdaLayerNormZero(dim, num_embeddings=None, norm_type="layer_norm")

    def forward(
        self,
        x: torch.Tensor,
        x_cond: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, dim) all channels
            x_cond: (B, H, W, dim_kv) conditional channels (extracted before embedding)
        Returns:
            (B, H, W, dim) cross-attended output
        """
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = self.adain(
            timestep=timestep, class_labels=class_labels, emb=emb
        )

        # Cross-attention: Q from x, K/V from x_cond
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + msa_scale[:, None, None]) + msa_shift[:, None, None]
        x_msa = self.attn(x_norm, x_cond)
        x_msa = x_msa * (1 + msa_gate[:, None, None])
        x = x + self.drop_path(x_msa)

        # MLP
        x_mlp = self.norm2(x)
        x_mlp = x_mlp * (1 + mlp_scale[:, None, None]) + mlp_shift[:, None, None]
        x_mlp = self.mlp(x_mlp)
        x_mlp = x_mlp * (1 + mlp_gate[:, None, None])
        x = x + self.drop_path(x_mlp)

        return x


class PDEBlock_SelfAttn(nn.Module):
    """
    Standard self-attention block for blocks after the first.
    Uses NATTEN for neighborhood self-attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        window_size: int = 7,
        stride: int = 1,
        use_pos_enc: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.dim = dim

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention2DTime_NATTEN(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=None,
            attn_drop=attn_drop,
            proj_drop=drop,
            resolution=window_size,
            use_pos_enc=use_pos_enc,
            stride=stride,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # DiT conditioning
        self.adain = AdaLayerNormZero(dim, num_embeddings=None, norm_type="layer_norm")

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, dim)
        Returns:
            (B, H, W, dim)
        """
        msa_shift, msa_scale, msa_gate, mlp_shift, mlp_scale, mlp_gate = self.adain(
            timestep=timestep, class_labels=class_labels, emb=emb
        )

        # Self-attention
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + msa_scale[:, None, None]) + msa_shift[:, None, None]
        x_msa = self.attn(x_norm)
        x_msa = x_msa * (1 + msa_gate[:, None, None])
        x = x + self.drop_path(x_msa)

        # MLP
        x_mlp = self.norm2(x)
        x_mlp = x_mlp * (1 + mlp_scale[:, None, None]) + mlp_shift[:, None, None]
        x_mlp = self.mlp(x_mlp)
        x_mlp = x_mlp * (1 + mlp_gate[:, None, None])
        x = x + self.drop_path(x_mlp)

        return x


class PDEStage_CrossThenSelf(nn.Module):
    """
    Stage with:
    - First block: cross-attention (Q from all, K/V from conditional)
    - Remaining blocks: self-attention
    
    Conditional channels are extracted before the embedding layer and passed
    to the first cross-attention block only.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        conditional_channels: Sequence[int],
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        use_pos_enc: Optional[str] = None,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conditional_channels = sorted(list(conditional_channels))
        
        # First block: cross-attention
        self.first_block = PDEBlock_CrossAttn(
            dim=dim,
            conditional_channels=conditional_channels,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            stride=stride,
        )
        
        # Remaining blocks: self-attention
        self.self_attn_blocks = nn.ModuleList([
            PDEBlock_SelfAttn(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                use_pos_enc=use_pos_enc,
                stride=stride,
            )
            for _ in range(depth - 1)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        x_cond: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, C, H, W) channels-first
            x_cond: (B, H_orig, W_orig, dim_kv) conditional channels from input (pre-embedding)
            cond: conditioning embedding
        """
        # (B, C, H, W) -> (B, H, W, C)
        x = hidden_states.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        
        # Resize x_cond to match current spatial resolution if needed
        if x_cond is not None:
            # x_cond is (B, H_orig, W_orig, dim_kv)
            H_orig, W_orig = x_cond.shape[1], x_cond.shape[2]
            if H != H_orig or W != W_orig:
                # Interpolate to match current resolution
                x_cond_resized = x_cond.permute(0, 3, 1, 2)  # (B, dim_kv, H_orig, W_orig)
                x_cond_resized = F.interpolate(x_cond_resized, size=(H, W), mode='bilinear', align_corners=False)
                x_cond_resized = x_cond_resized.permute(0, 2, 3, 1)  # (B, H, W, dim_kv)
            else:
                x_cond_resized = x_cond
        else:
            x_cond_resized = None
        
        # First block: cross-attention
        x = self.first_block(x, x_cond_resized, timestep=timestep, class_labels=class_labels, emb=cond)
        
        # Remaining blocks: self-attention
        for block in self.self_attn_blocks:
            x = block(x, timestep=timestep, class_labels=class_labels, emb=cond)
        
        # Back to channels-first
        return x.permute(0, 3, 1, 2)


class PDEImplCross(nn.Module):
    """
    Diffusion UNet model with cross-attention conditioning.
    
    Architecture:
    - Extract conditional channels from input before embedding
    - First block of each stage uses cross-attention (Q from all, K/V from conditional)
    - Remaining blocks use self-attention
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        window_size: int = 8,
        patch_size: Optional[int] = 4,
        hidden_size: int = 96,
        max_hidden_size: int = 2048,
        depth: List[int] = None,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        periodic: bool = True,
        conditional_channels: Sequence[int] = (),
        use_pos_enc: Optional[str] = None,
        natten_stride: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        if depth is None:
            depth = [2, 4, 4, 6, 4, 4, 2]
        assert len(depth) % 2 == 1, "Encoder and decoder depths must be equal."
        self.num_encoder_layers = len(depth) // 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.periodic = periodic
        self.max_hidden_size = max_hidden_size
        self.patch_size = patch_size
        self.conditional_channels = sorted(list(conditional_channels))

        StageClass = PDEStage_CrossThenSelf
        stage_args = {
            "drop_path": 0.0,
            "mlp_ratio": mlp_ratio,
            "conditional_channels": conditional_channels,
            "use_pos_enc": use_pos_enc,
            "stride": natten_stride,
        }

        if patch_size is not None:
            self.x_embedder = SimplePatchEmbed(in_channels, hidden_size, patch_size, bias=True)
            self.patch_size = patch_size
        else:
            self.x_embedder = OverlapPatchEmbed(in_channels, hidden_size, bias=True)
            self.patch_size = 1

        for i in range(self.num_encoder_layers + 1):
            hidden_size_layer = min(hidden_size * 2 ** i, max_hidden_size)
            setattr(self, f"t_embedder_{i}", TimestepEmbedder(hidden_size_layer))
            setattr(self, f"y_embedder_{i}", LabelEmbedder(num_classes, hidden_size_layer, class_dropout_prob))

        for i in range(self.num_encoder_layers):
            hidden_size_layer = min(hidden_size * 2 ** i, max_hidden_size)
            setattr(
                self,
                f"encoder_level_{i}",
                StageClass(
                    dim=hidden_size_layer,
                    num_heads=num_heads,
                    window_size=window_size,
                    depth=depth[i],
                    **stage_args,
                ),
            )
            keep_dim = hidden_size_layer == max_hidden_size
            setattr(self, f"down{i}_{i+1}", Downsample(hidden_size_layer, keep_dim=keep_dim))

        hidden_size_latent = min(hidden_size * 2 ** self.num_encoder_layers, max_hidden_size)
        self.latent = StageClass(
            dim=hidden_size_latent,
            num_heads=num_heads,
            window_size=window_size,
            depth=depth[self.num_encoder_layers],
            **stage_args,
        )

        hidden_size_layer0 = min(hidden_size * 2, max_hidden_size)
        keep_dim0 = hidden_size_layer0 >= max_hidden_size
        setattr(self, "up1_0", Upsample(hidden_size_layer0, keep_dim=keep_dim0))
        setattr(
            self,
            "reduce_chan_level0",
            nn.Conv2d(2 * min(hidden_size, max_hidden_size), hidden_size_layer0, kernel_size=1, bias=True),
        )
        setattr(
            self,
            "decoder_level_0",
            StageClass(
                dim=hidden_size_layer0,
                num_heads=num_heads,
                window_size=window_size,
                depth=depth[self.num_encoder_layers + 1],
                **stage_args,
            ),
        )

        for i in range(1, self.num_encoder_layers):
            hidden_size_layer = min(hidden_size * 2 ** i, max_hidden_size)
            if 2 * hidden_size_layer >= max_hidden_size:
                keep_dim = True
                hidden_size_upsample = max_hidden_size
            else:
                keep_dim = False
                hidden_size_upsample = 2 * hidden_size_layer

            setattr(self, f"up{i+1}_{i}", Upsample(hidden_size_upsample, keep_dim=keep_dim))
            setattr(
                self,
                f"reduce_chan_level{i}",
                nn.Conv2d(hidden_size_layer * 2, hidden_size_layer, kernel_size=1, bias=True),
            )
            setattr(
                self,
                f"decoder_level_{i}",
                StageClass(
                    dim=hidden_size_layer,
                    num_heads=num_heads,
                    window_size=window_size,
                    depth=depth[self.num_encoder_layers + i + 1],
                    **stage_args,
                ),
            )

        hidden_size_out = min(2 * hidden_size, max_hidden_size)
        self.output = nn.Conv2d(hidden_size_out, hidden_size_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.final_layer = FinalLayer(hidden_size_out, self.out_channels * self.patch_size * self.patch_size)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        for i in range(self.num_encoder_layers):
            nn.init.normal_(getattr(self, f"y_embedder_{i}").embedding_table.weight, std=0.02)
            nn.init.normal_(getattr(self, f"t_embedder_{i}").mlp[0].weight, std=0.02)
            nn.init.normal_(getattr(self, f"t_embedder_{i}").mlp[2].weight, std=0.02)

        # Initialize adain weights to zero for all blocks
        blocks = [getattr(self, f"encoder_level_{i}") for i in range(self.num_encoder_layers)]
        blocks += [self.latent]
        blocks += [getattr(self, f"decoder_level_{i}") for i in range(self.num_encoder_layers)]

        for stage in blocks:
            # First block (cross-attn)
            nn.init.constant_(stage.first_block.adain.linear.weight, 0)
            nn.init.constant_(stage.first_block.adain.linear.bias, 0)
            # Self-attn blocks
            for blc in stage.self_attn_blocks:
                nn.init.constant_(blc.adain.linear.weight, 0)
                nn.init.constant_(blc.adain.linear.bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.out_proj.weight, 0)
        nn.init.constant_(self.final_layer.out_proj.bias, 0)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor], y: Optional[torch.LongTensor]):
        # Extract conditional channels from input BEFORE embedding
        # x is (B, C, H, W), conditional_channels are indices into C dimension
        if len(self.conditional_channels) > 0:
            x_cond = x[:, self.conditional_channels, :, :]  # (B, len(cond_ch), H, W)
            x_cond = x_cond.permute(0, 2, 3, 1)  # (B, H, W, len(cond_ch))
        else:
            x_cond = None
        
        # Embed input
        x = self.x_embedder(x)

        if t is None:
            t = torch.tensor([0.0], device=x.device)
        if t.ndim == 0:
            t = t.unsqueeze(0).repeat(x.shape[0]).to(x.device)
        t = t * 1000.0

        if y is None:
            y = torch.ones(x.shape[0], dtype=torch.long, device=x.device) * self.num_classes

        emb_list = []
        for i in range(self.num_encoder_layers + 1):
            t_emb = getattr(self, f"t_embedder_{i}")(t)
            y_emb = getattr(self, f"y_embedder_{i}")(y, self.training)
            emb_list.append(t_emb + y_emb)

        residuals_list = []
        for i, c in enumerate(emb_list[:-1]):
            out_enc_level = getattr(self, f"encoder_level_{i}")(x, x_cond=x_cond, cond=c)
            residuals_list.append(out_enc_level)
            x = getattr(self, f"down{i}_{i+1}")(out_enc_level)

        c = emb_list[-1]
        x = self.latent(x, x_cond=x_cond, cond=c)

        for i, (residual, emb) in enumerate(zip(residuals_list[1:][::-1], emb_list[1:-1][::-1])):
            x = getattr(self, f"up{self.num_encoder_layers - i}_{self.num_encoder_layers - i - 1}")(x)
            x = torch.cat([x, residual], dim=1)
            x = getattr(self, f"reduce_chan_level{self.num_encoder_layers - i - 1}")(x)
            x = getattr(self, f"decoder_level_{self.num_encoder_layers - i - 1}")(x, x_cond=x_cond, cond=emb)

        x = getattr(self, "up1_0")(x)
        x = torch.cat([x, residuals_list[0]], dim=1)
        x = getattr(self, "reduce_chan_level0")(x)
        x = getattr(self, "decoder_level_0")(x, x_cond=x_cond, cond=emb_list[1])

        x = self.output(x)
        x = self.final_layer(x, emb_list[1])

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[:3] + (self.patch_size, self.patch_size, self.out_channels))
        h, w = x.shape[1], x.shape[2]
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(-1, self.out_channels, h * self.patch_size, w * self.patch_size)
        return x


@dataclass
class PDEOutput(BaseOutput):
    sample: torch.Tensor


class PDETransformerFA_Cross(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        sample_size: int,
        in_channels: int,
        out_channels: int,
        type: str,
        periodic: bool = False,
        window_size: int = 8,
        patch_size: Optional[int] = 4,
        conditional_channels: Sequence[int] = (0,),
        natten_stride: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        args = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "patch_size": patch_size,
            "periodic": periodic,
            "window_size": window_size,
            "conditional_channels": conditional_channels,
            "natten_stride": natten_stride,
        }
        args.update(kwargs)
        self.model: PDEImplCross = PDE_models_cross[type](**args)
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        output = self.model.forward(hidden_states, timestep, class_labels)
        if not return_dict:
            return (output,)
        return PDEOutput(sample=output)


def PDE_S_cross(**kwargs):
    return PDEImplCross(down_factor=2, hidden_size=96, num_heads=4, depth=[2, 5, 8, 5, 2], mlp_ratio=4, **kwargs)


def PDE_B_cross(**kwargs):
    return PDEImplCross(down_factor=2, hidden_size=192, num_heads=8, depth=[2, 5, 8, 5, 2], mlp_ratio=4, **kwargs)


def PDE_L_cross(**kwargs):
    return PDEImplCross(down_factor=2, hidden_size=384, num_heads=16, depth=[2, 5, 8, 5, 2], mlp_ratio=4, **kwargs)


PDE_models_cross = {
    "PDE-S": PDE_S_cross,
    "PDE-B": PDE_B_cross,
    "PDE-L": PDE_L_cross,
}
