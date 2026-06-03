import math

import torch
import torch.nn as nn
from einops import rearrange

from basicsr.models.archs.fusion_modules import CrossmodalAtten_imgeventalladd
from basicsr.models.archs.recurrent_sub_modules import ConvLayer, conv_down
from basicsr.models.archs.refid.XXNet_final_attenfusion_arch import FinalBidirectionAttenfusion


def _load_eamamba_block():
    try:
        from basicsr.models.archs.mamba.EAMamba.eamamba_block import EAMambaBlock
    except Exception:
        return None
    return EAMambaBlock


def sinusoidal_time_embedding(timesteps, dim):
    """Build sinusoidal embeddings for normalized scalar time values."""
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


class TemporalFallbackBlock(nn.Module):
    """Depthwise temporal conv fallback with the same [N, S, C] interface."""

    def __init__(self, channels, kernel_size=3):
        super(TemporalFallbackBlock, self).__init__()
        padding = kernel_size // 2
        self.temporal = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=padding,
            groups=channels
        )
        self.pointwise = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x):
        residual = x
        x = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + residual
        return x + self.pointwise(x)


class _TemporalEAMambaAdapter(nn.Module):
    """Adapter from [N, S, C] temporal tokens to EAMamba's [N, C, S, 1]."""

    def __init__(self, channels):
        super(_TemporalEAMambaAdapter, self).__init__()
        EAMambaBlock = _load_eamamba_block()
        if EAMambaBlock is None:
            raise ImportError("EAMambaBlock is unavailable.")
        self.block = EAMambaBlock(dim=channels)

    def forward(self, x):
        x = rearrange(x, "n s c -> n c s 1")
        x = self.block(x)
        return rearrange(x, "n c s 1 -> n s c")


class Ab_FusionBlock(nn.Module):
    """Time-query bi-temporal Mamba fusion over per-pixel feature trajectories."""

    def __init__(self, channels, num_heads=4, use_eamamba=True,
                 return_attention=False):
        super(Ab_FusionBlock, self).__init__()
        self.channels = channels
        self.return_attention = return_attention

        self.time_mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

        if use_eamamba:
            try:
                self.mamba_f = _TemporalEAMambaAdapter(channels)
                self.mamba_b = _TemporalEAMambaAdapter(channels)
            except Exception:
                self.mamba_f = TemporalFallbackBlock(channels)
                self.mamba_b = TemporalFallbackBlock(channels)
        else:
            self.mamba_f = TemporalFallbackBlock(channels)
            self.mamba_b = TemporalFallbackBlock(channels)

        self.temporal_proj = nn.Linear(channels * 2, channels)
        self.seq_norm = nn.LayerNorm(channels)
        self.query_norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=max(1, min(num_heads, channels)),
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.out_norm = nn.LayerNorm(channels)

    def forward(self, F_seq, t_seq, target_tau):
        if F_seq.dim() != 5:
            raise RuntimeError(
                f"Ab_FusionBlock expects [B,S,C,H,W], got {tuple(F_seq.shape)}."
            )

        b, s, c, h, w = F_seq.shape
        if c != self.channels:
            raise RuntimeError(
                f"Ab_FusionBlock was built for C={self.channels}, got C={c}."
            )

        t_seq = t_seq.to(device=F_seq.device, dtype=F_seq.dtype).reshape(s)
        target_tau = target_tau.to(device=F_seq.device, dtype=F_seq.dtype).reshape(-1)

        time_emb = sinusoidal_time_embedding(t_seq, c)
        time_emb = self.time_mlp(time_emb).view(1, s, c, 1, 1)
        F_seq = F_seq + time_emb

        tokens = rearrange(F_seq, "b s c h w -> (b h w) s c")
        tokens = self.seq_norm(tokens)

        forward_tokens = self.mamba_f(tokens)
        backward_tokens = torch.flip(
            self.mamba_b(torch.flip(tokens, dims=[1])),
            dims=[1],
        )
        z_seq = self.temporal_proj(torch.cat([forward_tokens, backward_tokens], dim=-1))
        z_seq = self.seq_norm(z_seq + tokens)

        query = sinusoidal_time_embedding(target_tau, c)
        query = self.query_mlp(query)
        query = self.query_norm(query)
        query = query.unsqueeze(0).expand(tokens.size(0), -1, -1)

        attn_out, attn_map = self.attn(
            query, z_seq, z_seq,
            need_weights=self.return_attention,
            average_attn_weights=False,
        )
        target_tokens = self.attn_norm(query + attn_out)
        target_tokens = self.out_norm(target_tokens + self.ffn(target_tokens))

        target = rearrange(
            target_tokens, "(b h w) t c -> b t c h w", b=b, h=h, w=w
        )
        aux = {
            "attn_map": attn_map if self.return_attention else None,
            "t_seq": t_seq.detach(),
            "target_tau": target_tau.detach(),
        }
        return target, aux


class SequenceFusionThenDownLayer(nn.Module):
    """Baseline encoder front-end followed by time-query fusion."""

    def __init__(self, in_channels, out_channels, num_heads=4, norm=None,
                 use_atten_fuse=False, use_eamamba=True,
                 return_attention=False):
        super(SequenceFusionThenDownLayer, self).__init__()
        self.use_atten_fuse = use_atten_fuse
        self.conv = ConvLayer(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1,
            relu_slope=0.2, norm=norm
        )
        self.relu = nn.LeakyReLU(0.2, inplace=False)
        if self.use_atten_fuse:
            self.atten_fuse = CrossmodalAtten_imgeventalladd(
                c=in_channels, c_out=out_channels, DW_Expand=1, FFN_Expand=2
            )
        self.down = conv_down(out_channels, out_channels, bias=False)
        self.fusion = Ab_FusionBlock(
            out_channels, num_heads=num_heads, use_eamamba=use_eamamba,
            return_attention=return_attention
        )
        self.last_aux = None

    def forward(self, x, image_feature=None, target_tau=None):
        b, t, _, _, _ = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        if image_feature is not None:
            image_feature = rearrange(
                image_feature[:, None].expand(-1, t, -1, -1, -1),
                "b t c h w -> (b t) c h w",
            )
            if self.use_atten_fuse:
                x = self.atten_fuse(x, image_feature)
            else:
                x = self.conv(x + image_feature)
                x = self.relu(x)
        else:
            x = self.conv(x)
            x = self.relu(x)

        x = self.down(x)
        x = rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)

        t_seq = torch.linspace(0.0, 1.0, steps=t, device=x.device, dtype=x.dtype)
        if target_tau is None:
            target_tau = t_seq
        x, self.last_aux = self.fusion(x, t_seq, target_tau)
        return x


class AbFusionBlock(FinalBidirectionAttenfusion):
    """REFID ablation using Ab_FusionBlock in the event/image fusion path."""

    def __init__(self, *args, fusion_num_heads=4, fusion_use_eamamba=True,
                 fusion_return_attention=False, **kwargs):
        super(AbFusionBlock, self).__init__(*args, **kwargs)

        self.target_tau = None
        self.encoders_backward = nn.ModuleList()
        self.encoders_forward = nn.ModuleList()
        self.event_encoders = nn.ModuleList()
        for input_size, output_size, encoder_index in zip(
            self.encoder_input_sizes, self.encoder_output_sizes, self.encoder_indexs
        ):
            self.event_encoders.append(
                SequenceFusionThenDownLayer(
                    input_size,
                    output_size,
                    num_heads=fusion_num_heads,
                    norm=self.norm,
                    use_atten_fuse=encoder_index == 1,
                    use_eamamba=fusion_use_eamamba,
                    return_attention=fusion_return_attention,
                )
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

        target_tau = torch.linspace(
            0.0, 1.0, steps=t, device=event.device, dtype=event.dtype
        )
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
