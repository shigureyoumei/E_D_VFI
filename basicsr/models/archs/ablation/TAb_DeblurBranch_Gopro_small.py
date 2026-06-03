import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from basicsr.models.archs.recurrent_sub_modules import (
    SimpleRecurrentThenDownAttenfusionmodifiedConvLayer,
)
from basicsr.models.archs.refid.XXNet_final_attenfusion_arch import (
    FinalBidirectionAttenfusion,
)

try:
    from basicsr.models.archs.mamba.EAMamba.eamamba_block import EAMambaBlock
except Exception:
    EAMambaBlock = None


def conv3x3(in_channels, out_channels, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
    )


class ResidualBlock(nn.Module):
    """Small residual block used by RGB and fallback event encoders."""

    def __init__(self, channels, relu_slope=0.2):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            conv3x3(channels, channels),
            nn.LeakyReLU(relu_slope, inplace=False),
            conv3x3(channels, channels),
        )

    def forward(self, x):
        return x + self.body(x)


class DepthwiseSeparableConv(nn.Module):
    """Lightweight local mixing for boundary/frequency correction."""

    def __init__(self, in_channels, out_channels, relu_slope=0.2):
        super(DepthwiseSeparableConv, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                bias=True,
            ),
            nn.LeakyReLU(relu_slope, inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        return self.body(x)


class RGBResEncoder(nn.Module):
    """RGB appearance encoder.

    Input:
        x: [B, 3, H, W]
    Output:
        f1: [B, C, H, W], f2: [B, 2C, H/2, W/2],
        f3: [B, 4C, H/4, W/4]
    """

    def __init__(self, in_channels=3, base_channels=32, num_blocks=2, relu_slope=0.2):
        super(RGBResEncoder, self).__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.head = nn.Sequential(
            conv3x3(in_channels, c1),
            nn.LeakyReLU(relu_slope, inplace=False),
        )
        self.stage1 = nn.Sequential(
            *[ResidualBlock(c1, relu_slope) for _ in range(num_blocks)]
        )
        self.down1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1, bias=False)
        self.stage2 = nn.Sequential(
            *[ResidualBlock(c2, relu_slope) for _ in range(num_blocks)]
        )
        self.down2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1, bias=False)
        self.stage3 = nn.Sequential(
            *[ResidualBlock(c3, relu_slope) for _ in range(num_blocks)]
        )

    def forward(self, x):
        f1 = self.stage1(self.head(x))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))
        return [f1, f2, f3]


class _FallbackEAMambaBlock(nn.Module):
    """Fallback keeps the interface modular when EAMamba is unavailable."""

    def __init__(self, dim, relu_slope=0.2, **kwargs):
        super(_FallbackEAMambaBlock, self).__init__()
        self.block = ResidualBlock(dim, relu_slope=relu_slope)

    def forward(self, x):
        return self.block(x)


def build_eamamba_block(channels, relu_slope=0.2, **kwargs):
    if EAMambaBlock is None:
        return _FallbackEAMambaBlock(channels, relu_slope=relu_slope)
    return EAMambaBlock(dim=channels, **kwargs)


class EventEAMambaEncoder(nn.Module):
    """Exposure-event encoder for boundary and blur-trajectory cues.

    Input:
        x: [B, C_evt, H, W]
    Output:
        f1: [B, C, H, W], f2: [B, 2C, H/2, W/2],
        f3: [B, 4C, H/4, W/4]
    """

    def __init__(
        self,
        in_channels=10,
        base_channels=32,
        relu_slope=0.2,
        scan_type="zigzag",
        scan_count=4,
        scan_merge_method="add",
        channel_mixer_type="Simple",
    ):
        super(EventEAMambaEncoder, self).__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        mamba_kwargs = dict(
            scan_type=scan_type,
            scan_count=scan_count,
            scan_merge_method=scan_merge_method,
            channel_mixer_type=channel_mixer_type,
        )
        self.head = nn.Sequential(
            conv3x3(in_channels, c1),
            nn.LeakyReLU(relu_slope, inplace=False),
        )
        self.stage1 = build_eamamba_block(c1, relu_slope=relu_slope, **mamba_kwargs)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1, bias=False)
        self.stage2 = build_eamamba_block(c2, relu_slope=relu_slope, **mamba_kwargs)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1, bias=False)
        self.stage3 = build_eamamba_block(c3, relu_slope=relu_slope, **mamba_kwargs)

    def forward(self, x):
        f1 = self.stage1(self.head(x))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))
        return [f1, f2, f3]


class EventAwareDeblurFusionBlock(nn.Module):
    """Event-selected, frequency-aware residual fusion.

    RGB remains the main representation. Events only select and modulate the
    high-frequency correction.
    """

    def __init__(
        self,
        channels,
        gate_type="spatial",
        use_selection_gate=True,
        use_frequency_correction=True,
        use_residual_gate=True,
        relu_slope=0.2,
    ):
        super(EventAwareDeblurFusionBlock, self).__init__()
        if gate_type not in ("spatial", "channel"):
            raise ValueError(f"gate_type must be 'spatial' or 'channel', got {gate_type}.")

        self.channels = channels
        self.gate_type = gate_type
        self.use_selection_gate = use_selection_gate
        self.use_frequency_correction = use_frequency_correction
        self.use_residual_gate = use_residual_gate

        if gate_type == "channel":
            self.selection_conv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1),
            )
        else:
            self.selection_conv = nn.Conv2d(
                channels * 2, 1, kernel_size=3, stride=1, padding=1
            )

        self.boundary_event = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, relu_slope=relu_slope),
            nn.LeakyReLU(relu_slope, inplace=False),
        )
        self.correction = nn.Sequential(
            conv3x3(channels * 3, channels),
            nn.LeakyReLU(relu_slope, inplace=False),
            ResidualBlock(channels, relu_slope=relu_slope),
            conv3x3(channels, channels),
        )
        self.residual_gate = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, f_rgb, f_evt):
        # f_rgb/f_evt: [B, C_s, H_s, W_s]
        if self.use_selection_gate:
            selection = torch.sigmoid(self.selection_conv(torch.cat([f_rgb, f_evt], dim=1)))
        else:
            gate_channels = self.channels if self.gate_type == "channel" else 1
            selection = f_evt.new_ones(
                f_evt.size(0),
                gate_channels,
                1 if self.gate_type == "channel" else f_evt.size(2),
                1 if self.gate_type == "channel" else f_evt.size(3),
            )
        f_evt_selected = selection * f_evt

        if self.use_frequency_correction:
            low_rgb = F.avg_pool2d(f_rgb, kernel_size=3, stride=1, padding=1)
            high_rgb = f_rgb - low_rgb
            boundary_evt = self.boundary_event(f_evt_selected)
            correction = self.correction(torch.cat([high_rgb, boundary_evt, f_rgb], dim=1))
        else:
            correction = f_evt.new_zeros(f_rgb.shape)

        if self.use_residual_gate:
            residual_gate = torch.sigmoid(self.residual_gate(f_evt_selected))
        else:
            residual_gate = f_evt.new_ones(f_rgb.shape)

        f_deblur = f_rgb + residual_gate * correction
        aux = {
            "selection": selection,
            "event_selected": f_evt_selected,
            "correction": correction,
            "residual_gate": residual_gate,
        }
        return f_deblur, aux


class EventAwareDeblurBranch(nn.Module):
    """Blur-aware feature candidate branch for one RGB blur and exposure voxel."""

    def __init__(
        self,
        rgb_in_channels=3,
        event_in_channels=10,
        base_channels=32,
        num_stages=3,
        gate_type="spatial",
        use_selection_gate=True,
        use_frequency_correction=True,
        use_residual_gate=True,
        num_rgb_blocks=2,
        relu_slope=0.2,
    ):
        super(EventAwareDeblurBranch, self).__init__()
        if num_stages != 3:
            raise ValueError("EventAwareDeblurBranch currently supports exactly 3 stages.")

        self.rgb_encoder = RGBResEncoder(
            in_channels=rgb_in_channels,
            base_channels=base_channels,
            num_blocks=num_rgb_blocks,
            relu_slope=relu_slope,
        )
        self.event_encoder = EventEAMambaEncoder(
            in_channels=event_in_channels,
            base_channels=base_channels,
            relu_slope=relu_slope,
        )
        channels = [base_channels, base_channels * 2, base_channels * 4]
        self.fusion_blocks = nn.ModuleList(
            [
                EventAwareDeblurFusionBlock(
                    c,
                    gate_type=gate_type,
                    use_selection_gate=use_selection_gate,
                    use_frequency_correction=use_frequency_correction,
                    use_residual_gate=use_residual_gate,
                    relu_slope=relu_slope,
                )
                for c in channels
            ]
        )

    def forward(self, rgb_blur, event_exp):
        # rgb_blur: [B, 3, H, W], event_exp: [B, C_evt, H, W]
        rgb_feats = self.rgb_encoder(rgb_blur)
        event_feats = self.event_encoder(event_exp)

        features = []
        aux = []
        for fusion, f_rgb, f_evt in zip(self.fusion_blocks, rgb_feats, event_feats):
            f_deblur, scale_aux = fusion(f_rgb, f_evt)
            features.append(f_deblur)
            aux.append(scale_aux)
        return features, aux


class AbDeblurBranch(FinalBidirectionAttenfusion):
    """REFID baseline with an event-aware deblur branch for image conditioning.

    The branch consumes the two endpoint blur images and their exposure event
    stacks from `x`:
        x = [left_rgb, left_exp_event, right_rgb, right_exp_event].
    It produces blur-aware features that replace the baseline image encoder
    features used by the recurrent event decoder.
    """

    def __init__(
        self,
        *args,
        gate_type="spatial",
        use_selection_gate=True,
        use_frequency_correction=True,
        use_residual_gate=True,
        deblur_event_chn=None,
        **kwargs,
    ):
        super(AbDeblurBranch, self).__init__(*args, **kwargs)

        if deblur_event_chn is None:
            deblur_event_chn = (self.img_chn - 6) // 2
        if deblur_event_chn <= 0:
            raise ValueError(
                "deblur_event_chn must be positive. Use return_deblur_voxel=true "
                "or pass deblur_event_chn explicitly."
            )
        self.deblur_event_chn = deblur_event_chn

        self.deblur_branch = EventAwareDeblurBranch(
            rgb_in_channels=3,
            event_in_channels=deblur_event_chn,
            base_channels=self.base_num_channels,
            num_stages=3,
            gate_type=gate_type,
            use_selection_gate=use_selection_gate,
            use_frequency_correction=use_frequency_correction,
            use_residual_gate=use_residual_gate,
        )

        c = self.base_num_channels
        self.pair_fuse = nn.ModuleList(
            [
                conv3x3(c * 2, c),
                conv3x3(c * 4, c * 2),
                conv3x3(c * 8, c * 4),
            ]
        )
        self.refid_head_from_deblur = nn.Sequential(
            conv3x3(c, c),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.refid_stage3_to_bottleneck = nn.Sequential(
            nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            ResidualBlock(c * 8, relu_slope=0.2),
        )

        # Rebuild event encoders after the parent initializer so the network is
        # independent of any baseline modules that concat image and event early.
        self.encoders_backward = nn.ModuleList()
        self.encoders_forward = nn.ModuleList()
        for input_size, output_size, encoder_index in zip(
            self.encoder_input_sizes, self.encoder_output_sizes, self.encoder_indexs
        ):
            use_atten_fuse = encoder_index == 1
            self.encoders_backward.append(
                SimpleRecurrentThenDownAttenfusionmodifiedConvLayer(
                    input_size,
                    output_size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    fuse_two_direction=False,
                    norm=self.norm,
                    num_block=kwargs.get("num_block", 3),
                    use_atten_fuse=use_atten_fuse,
                )
            )
            self.encoders_forward.append(
                SimpleRecurrentThenDownAttenfusionmodifiedConvLayer(
                    input_size,
                    output_size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    fuse_two_direction=True,
                    norm=self.norm,
                    num_block=kwargs.get("num_block", 3),
                    use_atten_fuse=use_atten_fuse,
                )
            )

    def _split_deblur_inputs(self, x):
        if x.dim() == 5:
            x = rearrange(x, "b t c h w -> b (t c) h w")
        if x.size(1) < 6 + 2 * self.deblur_event_chn:
            raise RuntimeError(
                f"Expected at least {6 + 2 * self.deblur_event_chn} channels in x, "
                f"got {x.size(1)}."
            )

        left_rgb = x[:, 0:3, :, :]
        left_event_start = 3
        left_event_end = left_event_start + self.deblur_event_chn
        right_rgb_start = left_event_end
        right_rgb_end = right_rgb_start + 3
        right_event_start = right_rgb_end
        right_event_end = right_event_start + self.deblur_event_chn

        left_event = x[:, left_event_start:left_event_end, :, :]
        right_rgb = x[:, right_rgb_start:right_rgb_end, :, :]
        right_event = x[:, right_event_start:right_event_end, :, :]
        return left_rgb, left_event, right_rgb, right_event

    def _build_deblur_condition(self, x):
        left_rgb, left_event, right_rgb, right_event = self._split_deblur_inputs(x)
        left_feats, left_aux = self.deblur_branch(left_rgb, left_event)
        right_feats, right_aux = self.deblur_branch(right_rgb, right_event)

        deblur_feats = [
            fuse(torch.cat([left_feat, right_feat], dim=1))
            for fuse, left_feat, right_feat in zip(self.pair_fuse, left_feats, right_feats)
        ]
        head = self.refid_head_from_deblur(deblur_feats[0])

        # Existing REFID decoder expects conditioning at H/2, H/4, and H/8.
        x_blocks = [
            deblur_feats[1],
            deblur_feats[2],
            self.refid_stage3_to_bottleneck(deblur_feats[2]),
        ]
        self.latest_deblur_aux = {"left": left_aux, "right": right_aux}
        return head, x_blocks

    def forward(self, x, event):
        head, x_blocks = self._build_deblur_condition(x)
        b, t, _, _, _ = event.shape

        event = rearrange(event, "b t c h w -> (b t) c h w")
        event = self.head(event)
        event = rearrange(event, "(b t) c h w -> b t c h w", b=b, t=t)

        out_l = []
        backward_all_states = []
        backward_prev_states = [None] * self.num_encoders
        forward_prev_states = [None] * self.num_encoders
        prev_states_decoder = [None] * self.num_encoders

        for frame_idx in range(t - 1, -1, -1):
            e_cur = event[:, frame_idx, :, :, :]
            for i, back_encoder in enumerate(self.encoders_backward):
                y = None if i == 0 else x_blocks[i - 1]
                e_cur, state = back_encoder(
                    x=e_cur, y=y, prev_state=backward_prev_states[i]
                )
                backward_prev_states[i] = state
            backward_all_states.insert(0, list(backward_prev_states))

        for frame_idx in range(t):
            e_blocks = []
            e_cur = event[:, frame_idx, :, :, :]
            for i, encoder in enumerate(self.encoders_forward):
                y = None if i == 0 else x_blocks[i - 1]
                e_cur, state = encoder(
                    x=e_cur,
                    y=y,
                    prev_state=forward_prev_states[i],
                    bi_direction_state=backward_all_states[frame_idx][i],
                )
                e_blocks.append(e_cur)
                forward_prev_states[i] = state

            for i, resblock in enumerate(self.resblocks):
                e_cur = resblock(e_cur + x_blocks[-1]) if i == 0 else resblock(e_cur)

            for i, decoder in enumerate(self.decoders):
                skip = e_blocks[self.num_encoders - i - 1]
                e_cur, state = decoder(
                    self.apply_skip_connection(e_cur, skip), prev_states_decoder[i]
                )
                prev_states_decoder[i] = state

            out_l.append(self.pred(self.apply_skip_connection(e_cur, head)))

        return torch.stack(out_l, dim=1)


if __name__ == "__main__":
    branch = EventAwareDeblurBranch(
        rgb_in_channels=3,
        event_in_channels=10,
        base_channels=32,
        gate_type="spatial",
    )
    rgb = torch.randn(2, 3, 64, 80)
    event_exp = torch.randn(2, 10, 64, 80)
    feats, aux = branch(rgb, event_exp)
    for feat in feats:
        print(tuple(feat.shape))
    print(aux[0].keys())
