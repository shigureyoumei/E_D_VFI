import torch
import torch.nn as nn
from einops import rearrange

from basicsr.models.archs.fusion_modules import CrossmodalAtten_imgeventalladd
from basicsr.models.archs.mamba.EAMamba.eamamba_block import EAMambaBlock
from basicsr.models.archs.recurrent_sub_modules import ConvLayer, conv_down
from basicsr.models.archs.refid.XXNet_final_attenfusion_arch import FinalBidirectionAttenfusion


class MambaEVRBlock(nn.Module):
    """Bidirectional temporal Mamba processing for a complete event sequence."""

    def __init__(self, channels, num_block=1):
        super(MambaEVRBlock, self).__init__()
        self.channels = channels
        block_count = max(num_block, 1)
        self.forward_blocks = nn.Sequential(
            *[EAMambaBlock(dim=channels) for _ in range(block_count)]
        )
        self.backward_blocks = nn.Sequential(
            *[EAMambaBlock(dim=channels) for _ in range(block_count)]
        )
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1)

    def _run_direction(self, blocks, x):
        b, t, c, h, w = x.shape
        temporal_maps = rearrange(x, "b t c h w -> (b h w) c t 1")
        temporal_maps = blocks(temporal_maps)
        return rearrange(
            temporal_maps, "(b h w) c t 1 -> b t c h w", b=b, h=h, w=w
        )

    def forward(self, x):
        if x.dim() != 5:
            raise RuntimeError(f"MambaEVRBlock expects [B,T,C,H,W], got {tuple(x.shape)}.")

        forward_features = self._run_direction(self.forward_blocks, x)
        backward_features = torch.flip(
            self._run_direction(self.backward_blocks, torch.flip(x, dims=[1])),
            dims=[1],
        )
        b, t, c, h, w = forward_features.shape
        gate = torch.sigmoid(
            self.gate(
                rearrange(
                    torch.cat([forward_features, backward_features], dim=2),
                    "b t c h w -> (b t) c h w",
                )
            )
        )
        gate = rearrange(gate, "(b t) c h w -> b t c h w", b=b, t=t)
        return gate * forward_features + (1.0 - gate) * backward_features


class SequenceMambaThenDownAttenfusionLayer(nn.Module):
    """Baseline event/image encoder front-end followed by full-sequence Mamba."""

    def __init__(self, in_channels, out_channels, num_block=1, norm=None,
                 use_atten_fuse=False):
        super(SequenceMambaThenDownAttenfusionLayer, self).__init__()
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
        self.mamba = MambaEVRBlock(out_channels, num_block=num_block)

    def forward(self, x, image_feature=None):
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
        return self.mamba(x)


class AbMambaBlock(FinalBidirectionAttenfusion):
    """REFID baseline whose encoder EVR propagation uses bidirectional Mamba."""

    def __init__(self, *args, **kwargs):
        num_block = kwargs.get("num_block", 3)
        super(AbMambaBlock, self).__init__(*args, **kwargs)

        # Drop the baseline recurrent encoders; keep all image/decoder modules.
        self.encoders_backward = nn.ModuleList()
        self.encoders_forward = nn.ModuleList()
        self.event_encoders = nn.ModuleList()
        for input_size, output_size, encoder_index in zip(
            self.encoder_input_sizes, self.encoder_output_sizes, self.encoder_indexs
        ):
            self.event_encoders.append(
                SequenceMambaThenDownAttenfusionLayer(
                    input_size,
                    output_size,
                    num_block=num_block,
                    norm=self.norm,
                    use_atten_fuse=encoder_index == 1,
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

        event_blocks = []
        for index, event_encoder in enumerate(self.event_encoders):
            fusion_feature = None if index == 0 else x_blocks[index - 1]
            event = event_encoder(event, fusion_feature)
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
