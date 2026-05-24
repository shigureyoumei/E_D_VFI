import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from basicsr.models.archs.recurrent_sub_modules import ConvLayer, UpsampleConvLayer, TransposedConvLayer, \
    RecurrentConvLayer, ResidualBlock, ConvLSTM, ConvGRU, ImageEncoderConvBlock, SimpleRecurrentConvLayer, SimpleRecurrentThenDownConvLayer, \
        TransposeRecurrentConvLayer, SimpleRecurrentThenDownAttenfusionConvLayer, SimpleRecurrentThenDownAttenfusionmodifiedConvLayer
from basicsr.models.archs.dcn_util import ModulatedDeformConvPack
from basicsr.models.archs.EAMamba.eamamba_block import EAMambaBlock
from basicsr.models.archs.event_guided_deblur_branch import SharedEventGuidedDeblurBranch
from basicsr.models.archs.motion_basis_interpolation_branch import MotionBasisInterpolationBranch
from basicsr.models.archs.fusion_modules import CrossmodalAtten_imgeventalladd
from einops import rearrange


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


def sinusoidal_embedding(time_ids, dim):
    half_dim = dim // 2
    if half_dim == 0:
        return time_ids[:, None]

    frequencies = torch.exp(
        torch.arange(half_dim, device=time_ids.device, dtype=time_ids.dtype)
        * -(torch.log(torch.tensor(10000.0, device=time_ids.device, dtype=time_ids.dtype)) / max(half_dim - 1, 1))
    )
    angles = time_ids[:, None] * frequencies[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = f.pad(emb, (0, 1))
    return emb


class TemporalEAMambaBlock(nn.Module):
    """Bidirectional temporal EAMamba over per-pixel event feature sequences."""

    def __init__(self, dim):
        super(TemporalEAMambaBlock, self).__init__()
        hidden_dim = dim * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.mamba_f = EAMambaBlock(dim=dim)
        self.mamba_b = EAMambaBlock(dim=dim)
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x):
        b, t, c, h, w = x.shape

        x_seq = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        time_ids = torch.linspace(0, 1, t, device=x.device, dtype=x.dtype)
        time_emb = self.time_mlp(sinusoidal_embedding(time_ids, c))
        x_seq = x_seq + time_emb[None, :, :]

        x_ea = x_seq.transpose(1, 2).unsqueeze(-1)
        y_f = self.mamba_f(x_ea).squeeze(-1).transpose(1, 2)
        y_b = torch.flip(
            self.mamba_b(torch.flip(x_ea, dims=[2])).squeeze(-1).transpose(1, 2),
            dims=[1],
        )

        y = self.proj(torch.cat([y_f, y_b], dim=-1))
        return y.reshape(b, h, w, t, c).permute(0, 3, 4, 1, 2)


class EAMambaThenDownAttenfusionConvLayer(nn.Module):
    """Event/image fusion, spatial downsample, then bidirectional temporal EAMamba."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 relu_slope=0.2, norm=None, use_atten_fuse=False):
        super(EAMambaThenDownAttenfusionConvLayer, self).__init__()
        self.relu_slope = relu_slope
        self.use_atten_fuse = use_atten_fuse

        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, relu_slope, norm)
        if relu_slope is not None:
            self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        if self.use_atten_fuse:
            self.atten_fuse = CrossmodalAtten_imgeventalladd(c=in_channels, c_out=out_channels, DW_Expand=1, FFN_Expand=2)

        self.temporal_mamba = TemporalEAMambaBlock(out_channels)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x, y=None):
        b, t, c, h, w = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        if y is not None:
            y = y[:, None, :, :, :].expand(-1, t, -1, -1, -1)
            y = rearrange(y, 'b t c h w -> (b t) c h w')
            if self.use_atten_fuse:
                x = self.atten_fuse(x, y)
            else:
                x = x + y
                x = self.conv(x)
                if self.relu_slope is not None:
                    x = self.relu(x)
        else:
            x = self.conv(x)
            if self.relu_slope is not None:
                x = self.relu(x)

        x = self.down(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', b=b, t=t)
        return self.temporal_mamba(x)


class FinalDecoderRecurrentUNet(nn.Module):
    def __init__(self, img_chn, ev_chn, out_chn=3, skip_type='sum', activation='sigmoid',
                 num_encoders=3, base_num_channels=32, num_residual_blocks=2, norm=None, use_recurrent_upsample_conv=True):
        super(FinalDecoderRecurrentUNet, self).__init__()

        self.ev_chn = ev_chn
        self.img_chn = img_chn
        self.out_chn = out_chn
        self.skip_type = skip_type
        self.apply_skip_connection = skip_sum if self.skip_type == 'sum' else skip_concat
        self.activation = activation
        self.norm = norm

        if use_recurrent_upsample_conv:
            print('Using Recurrent UpsampleConvLayer (slow, but recurrent in decoder)')
            self.UpsampleLayer = TransposeRecurrentConvLayer
        else:
            print('Using No recurrent UpsampleConvLayer (fast, but no recurrent in decoder)')
            self.UpsampleLayer = UpsampleConvLayer

        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        assert(self.ev_chn > 0)
        assert(self.img_chn > 0)
        assert(self.out_chn > 0)

        self.encoder_input_sizes = []
        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_indexs = []
        for i in range(self.num_encoders):
            self.encoder_indexs.append(i)

        self.encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]

        self.activation = getattr(torch, self.activation, 'sigmoid')

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=2, padding=0, norm=self.norm)) # kernei_size= 5, padidng =2 before

    def build_prediction_layer(self):
        self.pred = ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.out_chn, kernel_size=3, stride=1, padding=1, relu_slope=None, norm=self.norm)



class MambaMotionBidirectionalNetwork(FinalDecoderRecurrentUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.

    num_block: the number of blocks in each simpleconvlayer.
    """

    def __init__(self, img_chn, ev_chn, out_chn=3, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_recurrent_upsample_conv=True, num_block=3, use_first_dcn=False, use_reversed_voxel=False):
        super(MambaMotionBidirectionalNetwork, self).__init__(img_chn, ev_chn, out_chn, skip_type, activation,
                                                              num_encoders, base_num_channels,
                                                              num_residual_blocks, norm,
                                                              use_recurrent_upsample_conv)
        self.use_reversed_voxel = use_reversed_voxel

        if self.num_encoders != 3:
            raise ValueError('Branch-based MambaMotionBidirectionalNetwork currently expects num_encoders=3.')

        self.deblur_event_channels = max((self.img_chn - 6) // 2, 0)
        self.num_deblur_candidates = max(self.deblur_event_channels + 1, 1)
        deblur_branch_event_channels = max(self.deblur_event_channels, 1)
        self.num_motion_basis = 3

        self.deblur_branch = SharedEventGuidedDeblurBranch(
            event_channels=deblur_branch_event_channels,
            rgb_channels=3,
            base_channels=self.base_num_channels,
            num_rgb_blocks=2,
            relu_slope=0.2,
            num_candidates=self.num_deblur_candidates,
        )
        self.interpolation_branch = MotionBasisInterpolationBranch(
            feature_channels=(
                self.base_num_channels,
                self.base_num_channels * 2,
                self.base_num_channels * 4,
            ),
            event_channels=self.ev_chn,
            base_event_channels=self.base_num_channels,
            num_basis=self.num_motion_basis,
            use_eamamba=True,
        )
        self.candidate_deep_down = nn.Sequential(
            nn.Conv2d(self.base_num_channels * 4, self.max_num_channels,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
        )
        
        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

    def _split_blur_inputs(self, x):
        if x.dim() == 5:
            b0 = x[:, 0, :, :, :]
            b1 = x[:, -1, :, :, :]
            e0 = x.new_zeros(b0.size(0), max(self.deblur_event_channels, 1), b0.size(2), b0.size(3))
            e1 = x.new_zeros_like(e0)
            return b0, b1, e0, e1

        c_evt = self.deblur_event_channels
        b0 = x[:, 0:3, :, :]
        if c_evt > 0:
            e0 = x[:, 3:3 + c_evt, :, :]
            b1_start = 3 + c_evt
            b1 = x[:, b1_start:b1_start + 3, :, :]
            e1 = x[:, b1_start + 3:b1_start + 3 + c_evt, :, :]
        else:
            b1 = x[:, 3:6, :, :]
            e0 = x.new_zeros(x.size(0), 1, x.size(2), x.size(3))
            e1 = x.new_zeros_like(e0)
        return b0, b1, e0, e1

    def _select_between_events_and_tau(self, event):
        total_t = event.size(1)
        m = max(self.deblur_event_channels + 1, 1)

        event_forward = event
        forward_t = total_t
        if total_t % 2 == 0 and total_t // 2 > 2 * m:
            event_forward = event[:, :total_t // 2, :, :, :]
            forward_t = event_forward.size(1)

        n = forward_t - 2 * m
        if n > 0:
            e_between = event_forward[:, m:m + n, :, :, :]
            tau = torch.arange(1, n + 1, device=event.device, dtype=event.dtype) / (n + 1)
            return e_between, tau, m, n

        n = forward_t
        e_between = event_forward
        tau = torch.arange(1, n + 1, device=event.device, dtype=event.dtype) / (n + 1)
        return e_between, tau, 1, n

    def _endpoint_features_for_interpolation(self, f0_deblur, f1_deblur):
        # Use the sharp candidates nearest the inter-frame interval as interpolation anchors.
        return (
            tuple(scale_feat[:, -1, :, :, :] for scale_feat in f0_deblur),
            tuple(scale_feat[:, 0, :, :, :] for scale_feat in f1_deblur),
        )

    def _assemble_candidate_sequence(self, f0_deblur, f1_deblur, interp_features):
        sequence_features = []
        compact_features = []
        for f0_s, f1_s, interp_s in zip(f0_deblur, f1_deblur, interp_features):
            compact_features.append(torch.cat([
                f0_s,
                interp_s,
                f1_s,
            ], dim=1))
            sequence_features.append(torch.cat([
                f0_s,
                interp_s,
                f1_s,
            ], dim=1))
        return sequence_features, compact_features

    def _apply_2d_to_sequence(self, module, x):
        b, t, c, h, w = x.shape
        y = module(rearrange(x, 'b t c h w -> (b t) c h w'))
        _, c_out, h_out, w_out = y.shape
        return rearrange(y, '(b t) c h w -> b t c h w', b=b, t=t)

    def _decode_candidate_sequence(self, sequence_features):
        head_seq = sequence_features[0]
        block_0 = sequence_features[1]
        block_1 = sequence_features[2]
        block_2 = self._apply_2d_to_sequence(self.candidate_deep_down, block_1)
        e_blocks_all = [block_0, block_1, block_2]

        t = head_seq.size(1)
        out_l = []
        prev_states_decoder = [None] * self.num_encoders
        for frame_idx in range(t):
            e_blocks = [e_block[:, frame_idx, :, :, :] for e_block in e_blocks_all]
            e_cur = block_2[:, frame_idx, :, :, :]

            for resblock in self.resblocks:
                e_cur = resblock(e_cur)

            for i, decoder in enumerate(self.decoders):
                e_cur, state = decoder(self.apply_skip_connection(e_cur, e_blocks[self.num_encoders - i - 1]),
                                       prev_states_decoder[i])
                prev_states_decoder[i] = state

            out = self.pred(self.apply_skip_connection(e_cur, head_seq[:, frame_idx, :, :, :]))
            out_l.append(out)

        return torch.stack(out_l, dim=1)

    def forward(self, x, event):
        """
        x: [B, 6 + 2*(m-1), H, W], ordered as
           left_rgb, left_exposure_event, right_rgb, right_exposure_event.
        event: [B, 2*m+n, C_event, H, W] for Ruisi-style datasets.
        return: [B, 2*m+n, out_chn, H, W], compatible with the existing GT layout.
        """
        b0, b1, e_exp0, e_exp1 = self._split_blur_inputs(x)
        f0_deblur, f1_deblur = self.deblur_branch(b0, b1, e_exp0, e_exp1)
        f0_interp_anchor, f1_interp_anchor = self._endpoint_features_for_interpolation(f0_deblur, f1_deblur)

        e_between, tau, m, _ = self._select_between_events_and_tau(event)
        if f0_deblur[0].size(1) != m or f1_deblur[0].size(1) != m:
            raise RuntimeError(
                f'Deblur candidate count mismatch: left={f0_deblur[0].size(1)}, '
                f'right={f1_deblur[0].size(1)}, expected m={m}.'
            )
        interp_features, debug = self.interpolation_branch(f0_interp_anchor, f1_interp_anchor, e_between, tau)
        self.latest_branch_debug = debug

        sequence_features, compact_features = self._assemble_candidate_sequence(
            f0_deblur, f1_deblur, interp_features,
        )
        expected_frames = 2 * m + tau.numel()
        if sequence_features[0].size(1) != expected_frames:
            raise RuntimeError(
                f'Candidate sequence length mismatch: got {sequence_features[0].size(1)}, '
                f'expected {expected_frames}.'
            )
        self.latest_compact_candidate_features = compact_features
        return self._decode_candidate_sequence(sequence_features)


class EventGuidedDeblurCandidateNetwork(nn.Module):
    """Two-frame exposure-event-guided deblur feature candidate network.

    This network intentionally returns multi-scale features instead of images.
    It is separate from MambaMotionBidirectionalNetwork so existing REFID-style
    training configs keep their original behavior.
    """

    def __init__(self, ev_chn, img_chn=3, base_num_channels=32, num_rgb_blocks=2,
                 relu_slope=0.2, **kwargs):
        super(EventGuidedDeblurCandidateNetwork, self).__init__()
        self.deblur_branch = SharedEventGuidedDeblurBranch(
            event_channels=ev_chn,
            rgb_channels=img_chn,
            base_channels=base_num_channels,
            num_rgb_blocks=num_rgb_blocks,
            relu_slope=relu_slope,
        )

    def forward(self, b0, b1, e_exp0, e_exp1):
        return self.deblur_branch(b0, b1, e_exp0, e_exp1)


if __name__ == '__main__':
    import time

    model = MambaMotionBidirectionalNetwork(img_chn=26, ev_chn=2, num_encoders=3)
    device = 'cuda'
    x = torch.rand(1, 26, 256, 256).to(device)
    event = torch.rand(1, 50, 2, 256, 256).to(device)
    model = model.to(device)

    start_time = time.time()
    result = model(x, event)
    end_time = time.time()

    inference_time = end_time - start_time
    print('Inference time:{}'.format(inference_time))
    print('Output shape:{}'.format(result.shape))
