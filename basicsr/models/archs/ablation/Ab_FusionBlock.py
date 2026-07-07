import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from basicsr.models.archs.recurrent_sub_modules import (
    ConvLayer,
    SimpleRecurrentConv,
    conv_down,
)
from basicsr.models.archs.refid.XXNet_final_attenfusion_arch import (
    FinalBidirectionAttenfusion,
)


def sinusoidal_time_embedding(timesteps, dim):
    """Build sinusoidal embeddings for normalized continuous time values."""
    if timesteps.dim() != 1:
        timesteps = timesteps.reshape(-1)

    half_dim = dim // 2
    if half_dim == 0:
        return timesteps[:, None]

    exponent = -math.log(10000.0) * torch.arange(
        half_dim, device=timesteps.device, dtype=timesteps.dtype
    )
    exponent = exponent / max(half_dim - 1, 1)
    emb = timesteps[:, None] * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _valid_num_heads(channels, requested_heads):
    requested_heads = max(1, min(int(requested_heads), int(channels)))
    while channels % requested_heads != 0:
        requested_heads -= 1
    return max(1, requested_heads)


class EventToRGBCrossAttentionFusion(nn.Module):
    """Time-aware cross attention where event features query RGB condition."""

    def __init__(
        self,
        channels,
        num_heads=4,
        return_attention=False,
        use_time_embedding=True,
        use_cross_attention=True,
        use_gate=True,
        window_size=8,
    ):
        super(EventToRGBCrossAttentionFusion, self).__init__()
        self.channels = channels
        self.return_attention = return_attention
        self.use_time_embedding = use_time_embedding
        self.use_cross_attention = use_cross_attention
        self.use_gate = use_gate
        self.window_size = max(1, int(window_size))

        self.time_mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.query_norm = nn.LayerNorm(channels)
        self.key_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=_valid_num_heads(channels, num_heads),
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(channels)
        self.gate = nn.Linear(channels, channels)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def _time_embedding(self, target_tau, channels, device, dtype):
        target_tau = target_tau.to(device=device, dtype=dtype).reshape(-1)
        time_emb = sinusoidal_time_embedding(target_tau, channels)
        return self.time_mlp(time_emb)

    def forward(self, event_feat, rgb_feat, target_tau):
        """
        Args:
            event_feat: [B, T, C, H, W]
            rgb_feat: [B, C, H, W]
            target_tau: [T]
        """
        if event_feat.dim() != 5:
            raise RuntimeError(
                "EventToRGBCrossAttentionFusion expects event_feat "
                f"[B,T,C,H,W], got {tuple(event_feat.shape)}."
            )
        if rgb_feat.dim() != 4:
            raise RuntimeError(
                "EventToRGBCrossAttentionFusion expects rgb_feat "
                f"[B,C,H,W], got {tuple(rgb_feat.shape)}."
            )

        b, t, c, h, w = event_feat.shape
        if c != self.channels or rgb_feat.size(1) != c:
            raise RuntimeError(
                f"Channel mismatch: event C={c}, rgb C={rgb_feat.size(1)}, "
                f"module C={self.channels}."
            )
        if target_tau.numel() != t:
            raise RuntimeError(
                f"target_tau should have T={t} values, got {target_tau.numel()}."
            )

        event_seq = event_feat
        rgb_seq = rgb_feat[:, None].expand(-1, t, -1, -1, -1)
        if self.use_time_embedding:
            time_emb = self._time_embedding(
                target_tau, c, event_feat.device, event_feat.dtype
            )
            time_emb = time_emb.view(1, t, c, 1, 1)
            event_seq = event_seq + time_emb
            rgb_seq = rgb_seq + time_emb

        bt = b * t
        event_maps = rearrange(event_seq, "b t c h w -> (b t) c h w")
        rgb_maps = rearrange(rgb_seq, "b t c h w -> (b t) c h w")

        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h or pad_w:
            event_maps = F.pad(event_maps, (0, pad_w, 0, pad_h))
            rgb_maps = F.pad(rgb_maps, (0, pad_w, 0, pad_h))

        _, _, h_pad, w_pad = event_maps.shape
        num_h = h_pad // self.window_size
        num_w = w_pad // self.window_size
        event_tokens = rearrange(
            event_maps,
            "bt c (nh wh) (nw ww) -> (bt nh nw) (wh ww) c",
            wh=self.window_size,
            ww=self.window_size,
        )
        rgb_tokens = rearrange(
            rgb_maps,
            "bt c (nh wh) (nw ww) -> (bt nh nw) (wh ww) c",
            wh=self.window_size,
            ww=self.window_size,
        )

        if self.use_cross_attention:
            query = self.query_norm(event_tokens)
            key_value = self.key_norm(rgb_tokens)
            attn_out, attn_map = self.attn(
                query,
                key_value,
                key_value,
                need_weights=self.return_attention,
                average_attn_weights=False,
            )
        else:
            attn_out = rgb_tokens
            attn_map = None

        if self.use_gate:
            gate = torch.sigmoid(self.gate(event_tokens))
            fused_tokens = event_tokens + gate * attn_out
        else:
            fused_tokens = event_tokens + attn_out

        fused_tokens = self.attn_norm(fused_tokens)
        fused_tokens = fused_tokens + self.ffn(self.ffn_norm(fused_tokens))
        fused_maps = rearrange(
            fused_tokens,
            "(bt nh nw) (wh ww) c -> bt c (nh wh) (nw ww)",
            bt=bt,
            nh=num_h,
            nw=num_w,
            wh=self.window_size,
            ww=self.window_size,
        )
        fused_maps = fused_maps[:, :, :h, :w]
        fused = rearrange(fused_maps, "(b t) c h w -> b t c h w", b=b, t=t)

        aux = {
            "attn_map": attn_map if self.return_attention else None,
            "target_tau": target_tau.detach(),
            "window_size": self.window_size,
        }
        return fused, aux


class TimeAwareFusionThenDownLayer(nn.Module):
    """Event/RGB fusion front-end followed by REFID-style conv and downsample."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads=4,
        norm=None,
        return_attention=False,
        use_time_embedding=True,
        use_cross_attention=True,
        use_gate=True,
        window_size=8,
    ):
        super(TimeAwareFusionThenDownLayer, self).__init__()
        self.use_cross_attention = use_cross_attention
        self.fusion = EventToRGBCrossAttentionFusion(
            in_channels,
            num_heads=num_heads,
            return_attention=return_attention,
            use_time_embedding=use_time_embedding,
            use_cross_attention=use_cross_attention,
            use_gate=use_gate,
            window_size=window_size,
        )
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
        self.last_aux = None

    def forward(self, x, image_feature=None, target_tau=None):
        """
        Args:
            x: [B, T, C_in, H, W]
            image_feature: [B, C_in, H, W] or None
            target_tau: [T]
        Returns:
            [B, T, C_out, H/2, W/2]
        """
        if x.dim() != 5:
            raise RuntimeError(
                f"TimeAwareFusionThenDownLayer expects [B,T,C,H,W], got {tuple(x.shape)}."
            )

        b, t, _, _, _ = x.shape
        if target_tau is None:
            target_tau = torch.linspace(
                1.0 / (t + 1),
                t / (t + 1),
                steps=t,
                device=x.device,
                dtype=x.dtype,
            )

        if image_feature is not None and self.use_cross_attention:
            x, self.last_aux = self.fusion(x, image_feature, target_tau)
        elif image_feature is not None:
            x = x + image_feature[:, None].expand(-1, t, -1, -1, -1)
            self.last_aux = {"attn_map": None, "target_tau": target_tau.detach()}
        else:
            self.last_aux = {"attn_map": None, "target_tau": target_tau.detach()}

        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.conv(x)
        x = self.relu(x)
        x = self.down(x)
        return rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)


class TimeAwareRecurrentThenDownFusionLayer(nn.Module):
    """Baseline recurrent event encoder with EGACA replaced by event-to-RGB attention."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        relu_slope=0.2,
        norm=None,
        num_block=3,
        fuse_two_direction=False,
        use_first_dcn=False,
        use_atten_fuse=False,
        num_heads=4,
        return_attention=False,
        use_time_embedding=True,
        use_cross_attention=True,
        use_gate=True,
        window_size=8,
    ):
        super(TimeAwareRecurrentThenDownFusionLayer, self).__init__()
        if use_first_dcn:
            raise NotImplementedError(
                "use_first_dcn is not supported in AbFusionBlock fusion ablation."
            )
        self.relu_slope = relu_slope
        self.use_cross_attention = use_cross_attention
        self.use_time_embedding = use_time_embedding

        self.conv = ConvLayer(
            in_channels, out_channels, kernel_size, stride, padding,
            relu_slope, norm
        )
        if relu_slope is not None:
            self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        self.recurrent_block = SimpleRecurrentConv(
            out_channels, out_channels, num_block=num_block
        )
        if fuse_two_direction:
            self.fuse_two_dir = ConvLayer(
                2 * out_channels, out_channels, 1, 1, 0, relu_slope, norm
            )

        self.rgb_proj = ConvLayer(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1,
            relu_slope=relu_slope, norm=norm
        )
        self.fusion = EventToRGBCrossAttentionFusion(
            out_channels,
            num_heads=num_heads,
            return_attention=return_attention,
            use_time_embedding=use_time_embedding,
            use_cross_attention=use_cross_attention,
            use_gate=use_gate,
            window_size=window_size,
        )
        self.down = conv_down(out_channels, out_channels, bias=False)
        self.last_aux = None

    def forward(
        self,
        x,
        y=None,
        prev_state=None,
        bi_direction_state=None,
        target_tau=None,
    ):
        # 1) Preserve baseline event recurrent processing.
        x = self.conv(x)
        if self.relu_slope is not None:
            x = self.relu(x)
        x, state = self.recurrent_block(x, prev_state)

        if bi_direction_state is not None:
            x = torch.cat((x, bi_direction_state), 1)
            x = self.fuse_two_dir(x)

        # 2) Replace EGACA/image fusion: recurrent event is Q, RGB condition is K/V.
        if y is not None:
            rgb = self.rgb_proj(y)
            if target_tau is None:
                target_tau = torch.tensor([0.5], device=x.device, dtype=x.dtype)
            else:
                target_tau = target_tau.to(device=x.device, dtype=x.dtype).reshape(1)

            if self.use_cross_attention:
                x_seq, self.last_aux = self.fusion(
                    x[:, None], rgb, target_tau
                )
                x = x_seq[:, 0]
            else:
                x = x + rgb
                self.last_aux = {
                    "attn_map": None,
                    "target_tau": target_tau.detach(),
                }
        else:
            self.last_aux = {
                "attn_map": None,
                "target_tau": None if target_tau is None else target_tau.detach(),
            }

        x = self.down(x)
        return x, state


class AbFusionBlock(FinalBidirectionAttenfusion):
    """REFID baseline recurrent encoder with EGACA replaced by cross attention."""

    def __init__(
        self,
        *args,
        fusion_num_heads=4,
        fusion_return_attention=False,
        fusion_use_time_embedding=True,
        fusion_use_cross_attention=True,
        fusion_use_gate=True,
        fusion_window_size=8,
        fusion_use_eamamba=None,
        fusion_num_middle_frames=None,
        **kwargs,
    ):
        # fusion_use_eamamba and fusion_num_middle_frames are accepted for
        # compatibility with older options; this ablation does not use them.
        super(AbFusionBlock, self).__init__(*args, **kwargs)

        self.target_tau = None
        self.encoders_backward = nn.ModuleList()
        self.encoders_forward = nn.ModuleList()
        for input_size, output_size, encoder_index in zip(
            self.encoder_input_sizes,
            self.encoder_output_sizes,
            self.encoder_indexs,
        ):
            use_fusion = fusion_use_cross_attention and encoder_index >= 1
            self.encoders_backward.append(
                TimeAwareRecurrentThenDownFusionLayer(
                    input_size,
                    output_size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    fuse_two_direction=False,
                    num_heads=fusion_num_heads,
                    norm=self.norm,
                    num_block=kwargs.get("num_block", 3),
                    use_first_dcn=kwargs.get("use_first_dcn", False),
                    return_attention=fusion_return_attention,
                    use_time_embedding=fusion_use_time_embedding,
                    use_cross_attention=use_fusion,
                    use_gate=fusion_use_gate,
                    window_size=fusion_window_size,
                )
            )
            self.encoders_forward.append(
                TimeAwareRecurrentThenDownFusionLayer(
                    input_size,
                    output_size,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    fuse_two_direction=True,
                    num_heads=fusion_num_heads,
                    norm=self.norm,
                    num_block=kwargs.get("num_block", 3),
                    use_first_dcn=kwargs.get("use_first_dcn", False),
                    return_attention=fusion_return_attention,
                    use_time_embedding=fusion_use_time_embedding,
                    use_cross_attention=use_fusion,
                    use_gate=fusion_use_gate,
                    window_size=fusion_window_size,
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
        if x.dim() == 5:
            x = rearrange(x, "b t c h w -> b (t c) h w")
        b, t, _, _, _ = event.shape

        head = self.head_img(x)
        image_feature = head
        x_blocks = []
        for image_encoder in self.img_encoders:
            image_feature = image_encoder(image_feature)
            x_blocks.append(image_feature)

        event = rearrange(event, "b t c h w -> (b t) c h w")
        event = self.head(event)
        event = rearrange(event, "(b t) c h w -> b t c h w", b=b, t=t)

        target_tau = self._default_target_tau(t, event.device, event.dtype)
        backward_all_states = []
        backward_prev_states = [None] * self.num_encoders
        forward_prev_states = [None] * self.num_encoders
        decoder_states = [None] * self.num_encoders

        for frame_index in range(t - 1, -1, -1):
            event_cur = event[:, frame_index]
            frame_tau = target_tau[frame_index:frame_index + 1]
            for index, back_encoder in enumerate(self.encoders_backward):
                fusion_feature = None if index == 0 else x_blocks[index - 1]
                event_cur, state = back_encoder(
                    x=event_cur,
                    y=fusion_feature,
                    prev_state=backward_prev_states[index],
                    target_tau=frame_tau,
                )
                backward_prev_states[index] = state
            backward_all_states.insert(0, backward_prev_states)

        outputs = []
        for frame_index in range(t):
            event_blocks = []
            event_cur = event[:, frame_index]
            frame_tau = target_tau[frame_index:frame_index + 1]
            for index, encoder in enumerate(self.encoders_forward):
                fusion_feature = None if index == 0 else x_blocks[index - 1]
                event_cur, state = encoder(
                    x=event_cur,
                    y=fusion_feature,
                    prev_state=forward_prev_states[index],
                    bi_direction_state=backward_all_states[frame_index][index],
                    target_tau=frame_tau,
                )
                event_blocks.append(event_cur)
                forward_prev_states[index] = state

            feature = event_cur

            for resblock_index, resblock in enumerate(self.resblocks):
                if resblock_index == 0:
                    feature = resblock(feature + x_blocks[-1])
                else:
                    feature = resblock(feature)

            for index, decoder in enumerate(self.decoders):
                feature, decoder_state = decoder(
                    self.apply_skip_connection(
                        feature, event_blocks[self.num_encoders - index - 1]
                    ),
                    decoder_states[index],
                )
                decoder_states[index] = decoder_state

            outputs.append(self.pred(self.apply_skip_connection(feature, head)))

        return torch.stack(outputs, dim=1)


if __name__ == "__main__":
    model = AbFusionBlock(
        img_chn=26,
        ev_chn=2,
        num_encoders=3,
        base_num_channels=8,
        num_residual_blocks=1,
        num_block=1,
        fusion_num_heads=4,
        fusion_window_size=4,
    )
    x_in = torch.rand(1, 26, 32, 32)
    event_in = torch.rand(1, 3, 2, 32, 32)
    with torch.no_grad():
        y = model(x_in, event_in)
    print(tuple(y.shape))
