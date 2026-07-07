import torch
import torch.nn as nn
from einops import rearrange

from basicsr.models.archs.ablation.Ab_SpatialTemporalMambabloc import (
    SequenceSpatialTemporalMambaThenDownAttenfusionLayer,
)
from basicsr.models.archs.ablation.TAb_DeblurBranch_Gopro_small import AbDeblurBranch


class AbSTMDeblurBranch(AbDeblurBranch):
    """Deblur-branch conditioning with spatial-temporal Mamba event encoders."""

    def __init__(self, *args, **kwargs):
        num_block = kwargs.get("num_block", 3)
        super(AbSTMDeblurBranch, self).__init__(*args, **kwargs)

        self.encoders_backward = nn.ModuleList()
        self.encoders_forward = nn.ModuleList()
        self.event_encoders = nn.ModuleList()
        for input_size, output_size, encoder_index in zip(
            self.encoder_input_sizes, self.encoder_output_sizes, self.encoder_indexs
        ):
            self.event_encoders.append(
                SequenceSpatialTemporalMambaThenDownAttenfusionLayer(
                    input_size,
                    output_size,
                    num_block=num_block,
                    norm=self.norm,
                    use_atten_fuse=encoder_index == 1,
                )
            )

    def forward(self, x, event):
        head, x_blocks = self._build_deblur_condition(x)
        b, t, _, _, _ = event.shape

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
