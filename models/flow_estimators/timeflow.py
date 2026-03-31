from __future__ import annotations

from typing import Sequence, Optional

import torch
import torch.nn as nn
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.layers.utils import get_pool_layer

from .utils import FlowConv, LearnablePositionEmbedding, SinusoidalPositionEmbeddings
from ..builder import FLOW_ESTIMATORS


class BasicBlock(nn.Module):
    '''
    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: int | Sequence[int], number of output channels.
        kernel_size: Sequence[int], convolution kernel size.
        res_skip: whether to use residual skip connection. bool.
        down: whether to apply downsampling. bool.
        down_first: whether to apply downsampling before the convolutions. bool.
        conv_down: whether to use convolutional downsampling. bool.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
    '''

    def __init__(self,
                 spatial_dims: int,
                 in_channels: int,
                 out_channels: int | Sequence[int],
                 kernel_size: Sequence[int],
                 t_embed_dim: int,
                 adaptive_norm: bool = False,
                 down: bool = False,
                 down_first: bool = False,
                 conv_down: bool = False,
                 pool_name: tuple | str | None = ('max', {'kernel_size': 2}),
                 bias: bool = True,
                 norm_name: tuple | str = ('INSTANCE', {'affine': False}),
                 act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                 dropout: tuple | str | float | None = None, ):
        super().__init__()
        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size]
        self.num_convs = len(kernel_size)
        if isinstance(out_channels, int):
            out_channels = [out_channels] * self.num_convs
        self.spatial_dims = spatial_dims
        self.down_first = down_first
        self.adaptive_norm = adaptive_norm
        self.convs = nn.Sequential()
        for i in range(self.num_convs):
            self.convs.append(
                # CONV - NORM - DROPOUT - ACT
                get_conv_layer(
                    spatial_dims,
                    in_channels if i == 0 else out_channels[i - 1],
                    out_channels[i],
                    kernel_size=kernel_size[i],
                    stride=1,
                    # adn_ordering="NDA",
                    dropout=dropout,
                    act=act_name,
                    norm=norm_name,
                    conv_only=False,
                    bias=bias,
                )
            )
        self.down = None
        if down:
            if conv_down:
                self.down = get_conv_layer(
                    spatial_dims,
                    in_channels if self.down_first else out_channels[-1],
                    in_channels if self.down_first else out_channels[-1],
                    kernel_size=3,
                    stride=2,
                    dropout=dropout,
                    act=act_name,
                    norm=norm_name,
                    conv_only=False,
                )
            else:
                self.down = get_pool_layer(pool_name, spatial_dims)

        if self.adaptive_norm:
            self.time_mlp = nn.Linear(t_embed_dim, out_channels[-1] * 2, bias=True)
        else:
            self.time_mlp = nn.Linear(t_embed_dim, out_channels[-1], bias=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''
        Args:
             x (torch.Tensor): input tensor, shape (B,C,H,W,[D]).
             t (torch.Tensor): time indices, float, shape (B, t_embed_dim).
        '''
        if (self.down is not None) and self.down_first:
            x = self.down(x)
        for conv in self.convs:
            x = conv(x)

        t_embed = self.time_mlp(t)
        spatial_shape = [1] * self.spatial_dims
        if self.adaptive_norm:
            gamma, beta = t_embed.chunk(2, dim=-1)
            gamma = gamma.view(*gamma.shape, *spatial_shape)
            beta = beta.view(*beta.shape, *spatial_shape)
            x = x * gamma + beta
        else:
            x = x + t_embed.view(*t_embed.shape, *spatial_shape)

        if (self.down is not None) and (not self.down_first):
            x = self.down(x)
        return x


class UpBlock(BasicBlock):
    def __init__(self,
                 spatial_dims: int,
                 in_channels: int,
                 skip_channels: int,
                 out_channels: int | Sequence[int],
                 kernel_size: Sequence[int],
                 t_embed_dim: int,
                 adaptive_norm: bool = False,
                 bias: bool = True,
                 up_transp_conv: bool = False,
                 upsample_kernel_size: int = 2,
                 transp_bias: bool = False,
                 norm_name: tuple | str = ('INSTANCE', {'affine': False}),
                 act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                 dropout: tuple | str | float | None = None,
                 ):
        super().__init__(spatial_dims,
                         in_channels + skip_channels, out_channels,
                         kernel_size, t_embed_dim, adaptive_norm,
                         False, False, False, None,
                         bias, norm_name, act_name, dropout)
        if in_channels > 0:
            if up_transp_conv:
                upsample_stride = upsample_kernel_size
                self.upsample = get_conv_layer(
                    spatial_dims,
                    in_channels,
                    in_channels,
                    kernel_size=upsample_kernel_size,
                    stride=upsample_stride,
                    conv_only=True,
                    is_transposed=True,
                    bias=transp_bias,
                )
            else:
                if spatial_dims == 2:
                    mode = 'bilinear'
                elif spatial_dims == 3:
                    mode = 'trilinear'
                else:
                    raise KeyError(f'Unsupported spatial dimension for Upsample:{spatial_dims}.')
                self.upsample = nn.Upsample(scale_factor=upsample_kernel_size,
                                            mode=mode,
                                            align_corners=True)

    def forward(self, inp: Optional[torch.Tensor], x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if inp is not None:
            inp = self.upsample(inp)
            x = torch.cat([inp, x], dim=1)
        for conv in self.convs:
            x = conv(x)

        t_embed = self.time_mlp(t)
        spatial_shape = [1] * self.spatial_dims
        if self.adaptive_norm:
            gamma, beta = t_embed.chunk(2, dim=-1)
            gamma = gamma.view(*gamma.shape, *spatial_shape)
            beta = beta.view(*beta.shape, *spatial_shape)
            x = x * gamma + beta
        else:
            x = x + t_embed.view(*t_embed.shape, *spatial_shape)

        return x


class CNNEncoder(nn.Module):
    def __init__(self,
                 spatial_dims: int,
                 down: bool,
                 in_chan: int,
                 out_channels: Sequence[int],
                 out_indices: Sequence[int],
                 block_config: dict,
                 ):
        '''
            block_config = dict(
                kernel_size=[3, 3],
                t_embed_dim=8,
                adaptive_norm=False,
                down_first=False,
                conv_down=True,
                bias=True,
                pool_name=('max', {'kernel_size': 2}),
                norm_name=('BatchNorm', {'affine': False if adaptive_norm else True}),
                act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                dropout=None,)
        '''
        super().__init__()
        self.num_levels = len(out_channels)
        self.encoder = nn.ModuleList()
        for i in range(self.num_levels):
            self.encoder.append(
                BasicBlock(
                    spatial_dims=spatial_dims,
                    in_channels=in_chan if i == 0 else out_channels[i - 1],
                    out_channels=out_channels[i],
                    down=False if i == 0 else down,
                    **block_config,
                )
            )
        self.out_indices = out_indices

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Sequence[torch.Tensor]:
        outs = []
        for i, layer in enumerate(self.encoder):
            x = layer(x, t)
            if i in self.out_indices:
                outs.append(x)
        return outs


class CNNDecoder(nn.Module):
    def __init__(self,
                 spatial_dims: int,
                 skip_channels: Sequence[int],
                 out_channels: Sequence[int],
                 block_config: dict,
                 ):
        '''
            block_config=dict(
                kernel_size=[3, 3],
                t_embed_dim=8,
                adaptive_norm=False,
                up_transp_conv=True,
                upsample_kernel_size=3,
                norm_name=('BatchNorm', {'affine': False if adaptive_norm else True}),
                act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                dropout=None,)
        '''
        super().__init__()
        self.num_levels = len(out_channels)
        self.decoder = nn.ModuleList()
        for i in range(self.num_levels):
            self.decoder.append(
                UpBlock(
                    spatial_dims=spatial_dims,
                    in_channels=0 if i == 0 else out_channels[i - 1],
                    skip_channels=skip_channels[i],
                    out_channels=out_channels[i],
                    **block_config,
                )
            )

    def forward(self, skips: Sequence[torch.Tensor], t: torch.Tensor) -> torch.Tensor:
        prev_dec = None
        for i, layer in enumerate(self.decoder):
            prev_dec = layer(prev_dec, skips[i], t)
        return prev_dec


@FLOW_ESTIMATORS.register_module()
class TimeFlow(nn.Module):
    def __init__(self,
                 t_embed_dim: int,
                 encoder_cfg,
                 decoder_cfg,
                 remain_cfg,
                 pe_type: str = 'spe',
                 max_periods: int = 100,
                 ):
        super().__init__()
        encoder_cfg.block_config.update(dict(t_embed_dim=t_embed_dim))
        decoder_cfg.block_config.update(dict(t_embed_dim=t_embed_dim))
        remain_cfg.block_config.update(dict(t_embed_dim=t_embed_dim))
        if pe_type == 'spe':
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(t_embed_dim, max_periods=max_periods),
                nn.Linear(t_embed_dim, t_embed_dim, bias=True),
                nn.SiLU()
            )
        elif pe_type == 'lpe':
            self.time_mlp = nn.Sequential(
                LearnablePositionEmbedding(t_embed_dim),
                nn.Linear(t_embed_dim, t_embed_dim, bias=True),
                nn.SiLU()
            )
        else:
            raise ValueError(f"PositionalEmbedding type pe_type should be 'spe' or 'lpe'. {pe_type} not implemented.")
        self.encoder = CNNEncoder(**encoder_cfg)
        self.decoder = CNNDecoder(**decoder_cfg)
        self.remain = CNNEncoder(**remain_cfg)
        self.spatial_dims = remain_cfg.spatial_dims
        self.flow_conv = FlowConv(
            spatial_dims=remain_cfg.spatial_dims,
            in_channels=remain_cfg.out_channels[-1], 
            kernel_size=3)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            src (torch.Tensor): source image, shape (B, C, D, H, W)
            tgt (torch.Tensor): target image, shape (B, C, D, H, W)
            t (torch.Tensor): time indices, float, shape (B,)
        '''
        t_embed = self.time_mlp(t)
        feats = self.encoder(torch.cat([src, tgt], dim=1), t_embed)
        feats = feats[::-1]
        dec = self.decoder(feats, t_embed)
        dec = self.remain(dec, t_embed)[-1]
        flow = self.flow_conv(dec)
        t_scaled = t.view(t.shape[0], *([1] * (self.spatial_dims + 1)))
        return t_scaled * flow
