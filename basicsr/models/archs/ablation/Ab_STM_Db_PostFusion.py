import os
import sys

import torch
import torch.nn as nn
from einops import rearrange

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from basicsr.models.archs.ablation.Ab_FusionBlock import (
    EventToRGBCrossAttentionFusion,
)
from basicsr.models.archs.ablation.Ab_SpatialTemporalMambabloc import (
    SpatialTemporalMambaBlock,
)
from basicsr.models.archs.ablation.TAb_DeblurBranch_Gopro_small import (
    AbDeblurBranch,
)
from basicsr.models.archs.recurrent_sub_modules import ConvLayer, conv_down


class SequenceSTMMambaThenPostFusionLayer(nn.Module):
    """Event STMamba encoder with post-Mamba deblur/RGB QKV fusion."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_block=1,
        norm=None,
        use_post_fusion=True,
        fusion_num_heads=4,
        fusion_return_attention=False,
        fusion_use_time_embedding=True,
        fusion_use_gate=True,
        fusion_window_size=8,
    ):
        super(SequenceSTMMambaThenPostFusionLayer, self).__init__()
        self.use_post_fusion = use_post_fusion
        self.conv = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            relu_slope=0.2,
            norm=norm,
        )
        self.relu = nn.LeakyReLU(0.2, inplace=False)
        self.down = conv_down(out_channels, out_channels, bias=False)
        self.mamba = SpatialTemporalMambaBlock(out_channels, num_block=num_block)

        self.rgb_proj = nn.Sequential(
            ConvLayer(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                relu_slope=0.2,
                norm=norm,
            ),
            conv_down(out_channels, out_channels, bias=False),
        )
        self.post_fusion = EventToRGBCrossAttentionFusion(
            out_channels,
            num_heads=fusion_num_heads,
            return_attention=fusion_return_attention,
            use_time_embedding=fusion_use_time_embedding,
            use_cross_attention=True,
            use_gate=fusion_use_gate,
            window_size=fusion_window_size,
        )
        self.last_aux = None

    def forward(self, x, image_feature=None, target_tau=None):
        """
        Args:
            x: [B, T, C_in, H, W]
            image_feature: [B, C_in, H, W] or None
            target_tau: [T]
        """
        if x.dim() != 5:
            raise RuntimeError(
                "SequenceSTMMambaThenPostFusionLayer expects [B,T,C,H,W], "
                f"got {tuple(x.shape)}."
            )

        b, t, _, _, _ = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.conv(x)
        x = self.relu(x)
        x = self.down(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)

        # First model the event sequence with STMamba; do not inject the shared
        # deblur/RGB condition before temporal-spatial propagation.
        x = self.mamba(x)

        if image_feature is not None and self.use_post_fusion:
            if target_tau is None:
                target_tau = torch.linspace(
                    1.0 / (t + 1),
                    t / (t + 1),
                    steps=t,
                    device=x.device,
                    dtype=x.dtype,
                )
            rgb_feature = self.rgb_proj(image_feature)
            x, self.last_aux = self.post_fusion(x, rgb_feature, target_tau)
        else:
            self.last_aux = {
                "attn_map": None,
                "target_tau": None if target_tau is None else target_tau.detach(),
            }
        return x


class AbSTMDeblurPostFusionBranch(AbDeblurBranch):
    """DeblurBranch + STMamba event encoder + post-STMamba QKV fusion."""

    def __init__(
        self,
        *args,
        fusion_num_heads=4,
        fusion_return_attention=False,
        fusion_use_time_embedding=True,
        fusion_use_gate=True,
        fusion_window_size=8,
        **kwargs,
    ):
        num_block = kwargs.get("num_block", 3)
        super(AbSTMDeblurPostFusionBranch, self).__init__(*args, **kwargs)

        self.target_tau = None
        self.encoders_backward = nn.ModuleList()
        self.encoders_forward = nn.ModuleList()
        self.event_encoders = nn.ModuleList()
        for input_size, output_size, encoder_index in zip(
            self.encoder_input_sizes,
            self.encoder_output_sizes,
            self.encoder_indexs,
        ):
            self.event_encoders.append(
                SequenceSTMMambaThenPostFusionLayer(
                    input_size,
                    output_size,
                    num_block=num_block,
                    norm=self.norm,
                    use_post_fusion=encoder_index >= 1,
                    fusion_num_heads=fusion_num_heads,
                    fusion_return_attention=fusion_return_attention,
                    fusion_use_time_embedding=fusion_use_time_embedding,
                    fusion_use_gate=fusion_use_gate,
                    fusion_window_size=fusion_window_size,
                )
            )

    def _default_target_tau(self, t, device, dtype):
        if self.target_tau is not None:
            return self.target_tau.to(device=device, dtype=dtype).reshape(t)
        return torch.linspace(
            1.0 / (t + 1),
            t / (t + 1),
            steps=t,
            device=device,
            dtype=dtype,
        )

    def forward(self, x, event):
        head, x_blocks = self._build_deblur_condition(x)
        b, t, _, _, _ = event.shape

        event = rearrange(event, "b t c h w -> (b t) c h w")
        event = self.head(event)
        event = rearrange(event, "(b t) c h w -> b t c h w", b=b, t=t)

        target_tau = self._default_target_tau(t, event.device, event.dtype)
        event_blocks = []
        for index, event_encoder in enumerate(self.event_encoders):
            fusion_feature = None if index == 0 else x_blocks[index - 1]
            event = event_encoder(event, fusion_feature, target_tau=target_tau)
            event_blocks.append(event)

        outputs = []
        decoder_states = [None] * self.num_encoders
        for frame_index in range(t):
            frame_blocks = [block[:, frame_index] for block in event_blocks]
            feature = frame_blocks[-1]
            for resblock_index, resblock in enumerate(self.resblocks):
                if resblock_index == 0:
                    feature = resblock(feature + x_blocks[-1])
                else:
                    feature = resblock(feature)

            for index, decoder in enumerate(self.decoders):
                feature, decoder_state = decoder(
                    self.apply_skip_connection(
                        feature, frame_blocks[self.num_encoders - index - 1]
                    ),
                    decoder_states[index],
                )
                decoder_states[index] = decoder_state

            outputs.append(self.pred(self.apply_skip_connection(feature, head)))

        return torch.stack(outputs, dim=1)


if __name__ == "__main__":
    model = AbSTMDeblurPostFusionBranch(
        img_chn=26,
        ev_chn=2,
        num_encoders=3,
        base_num_channels=8,
        num_block=1,
        num_residual_blocks=1,
        deblur_event_chn=10,
        fusion_num_heads=4,
        fusion_window_size=4,
    )
    x_in = torch.rand(1, 26, 32, 32)
    event_in = torch.rand(1, 3, 2, 32, 32)
    with torch.no_grad():
        y = model(x_in, event_in)
    print(tuple(y.shape))
