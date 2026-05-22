# basicsr/models/archs/EAMamba/eamamba_block.py

import torch
import torch.nn as nn

from .eamamba import MambaFormerBlock
from .module.scan import ScanTransform


"""
用法示例

from basicsr.models.archs.EAMamba.eamamba_block import EAMambaBlock

self.event_mamba = EAMambaBlock(
    dim=64,
    scan_type="zigzag",
    scan_count=4,
    scan_merge_method="add",
    channel_mixer_type="Simple",
)

x = self.event_mamba(x)

输入输出
[B, 64, H, W] -> [B, 64, H, W]

"""



class EAMambaBlock(nn.Module):
    """
    EAMamba block wrapper for intermediate feature processing.

    Input : [B, C, H, W]
    Output: [B, C, H, W]
    """
    def __init__(
        self,
        dim,
        scan_type="zigzag",
        scan_count=4,
        scan_merge_method="add",
        d_state=16,
        d_conv=4,
        expand=2,
        ffn_expansion_factor=2.0,
        bias=False,
        layernorm_type="WithBias",
        channel_mixer_type="Simple",
        use_checkpoint=False,
        conv_2d=False,
    ):
        super().__init__()

        scan_transform = ScanTransform(
            scan_type=scan_type,
            scan_count=scan_count,
            merge_method=scan_merge_method,
        )

        mamba_cfg = dict(
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            conv_2d=conv_2d,
            disable_z_branch=False,
            scan_type=scan_type,
            scan_count=scan_count,
            scan_merge_method=scan_merge_method,
        )

        self.block = MambaFormerBlock(
            dim=dim,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            layernorm_type=layernorm_type,
            scan_transform=scan_transform,
            mamba_cfg=mamba_cfg,
            use_checkpoint=use_checkpoint,
            channel_mixer_type=channel_mixer_type.lower(),
        )

    def forward(self, x):
        return self.block(x)