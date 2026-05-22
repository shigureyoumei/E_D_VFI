import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from basicsr.models.archs.recurrent_sub_modules import ConvLayer, UpsampleConvLayer, TransposedConvLayer, \
    RecurrentConvLayer, ResidualBlock, ConvLSTM, ConvGRU, ImageEncoderConvBlock, SimpleRecurrentConvLayer, SimpleRecurrentThenDownConvLayer, \
        TransposeRecurrentConvLayer, SimpleRecurrentThenDownAttenfusionConvLayer, SimpleRecurrentThenDownAttenfusionmodifiedConvLayer
from basicsr.models.archs.dcn_util import ModulatedDeformConvPack
from basicsr.models.archs.EAMamba.eamamba_block import EAMambaBlock
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
        ## event
        self.head = ConvLayer(self.ev_chn, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2, relu_slope=0.2)  # N x C x H x W -> N x 32 x H x W
        self.event_encoders = nn.ModuleList()

        for input_size, output_size, encoder_index in zip(self.encoder_input_sizes, self.encoder_output_sizes, self.encoder_indexs):
            # print('DEBUG: input size:{}'.format(input_size))
            # print('DEBUG: output size:{}'.format(output_size))
            print('Using temporal EAMamba event encoder!')
            use_atten_fuse = True if encoder_index == 1 else False
            self.event_encoders.append(EAMambaThenDownAttenfusionConvLayer(input_size, output_size,
                                                    kernel_size=3, stride=1, padding=1,
                                                    norm=self.norm, use_atten_fuse=use_atten_fuse))

        ## img
        self.head_img = ConvLayer(self.img_chn, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2, relu_slope=0.2)  # N x C x H x W -> N x 32 x H x W
        self.img_encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.img_encoders.append(ImageEncoderConvBlock(in_size=input_size, out_size=output_size,
                                                            downsample=True, relu_slope=0.2))
        
        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

    def forward(self, x, event):
        """
        :param x: b 2 c h w -> b, 2c, h, w
        :param event: b, t, num_bins, h, w -> b*t num_bins(2) h w 
        :return: b, t, out_chn, h, w

        One direction propt version
        TODO:  use_reversed_voxel!!!
        """
        # reshape
        if x.dim()==5:
            x = rearrange(x, 'b t c h w -> b (t c) h w') # sharp
        b, t, num_bins, h, w = event.size()
        event = rearrange(event, 'b t c h w -> (b t) c h w')

        
        # head
        x = self.head_img(x) # image feat
        head = x
        e = self.head(event)   # event feat
        # image encoder
        x_blocks = []
        for i, img_encoder in enumerate(self.img_encoders):
            x = img_encoder(x)
            x_blocks.append(x)

########
        ## temporal EAMamba event encoder
        e = rearrange(e, '(b t) c h w -> b t c h w', b=b, t=t)
        # if self.use_reversed_voxel:
        #     voxel, reversed_voxel = e.chunk(2,dim=1)
        #     t = t//2
        # else:
        #     voxel, reversed_voxel = e

        out_l = []
        prev_states_decoder = [None] * self.num_encoders
        e_blocks_all = []
        e_cur_all = e
        for i, event_encoder in enumerate(self.event_encoders):
            if i == 0:
                e_cur_all = event_encoder(e_cur_all, y=None)
            else:
                e_cur_all = event_encoder(e_cur_all, y=x_blocks[i - 1])
            e_blocks_all.append(e_cur_all)

        ## forward propt 
        for frame_idx in range(0,t):
            e_blocks = [e_block[:, frame_idx, :, :, :] for e_block in e_blocks_all] # skip feats for each frame
            e_cur = e_cur_all[:, frame_idx, :, :, :] # b,c,h,w

            ### add this!
            # residual blocks
            for i in range(len(self.resblocks)):
                if i == 0:
                    e_cur = self.resblocks[i](e_cur+x_blocks[-1])
                else:
                    e_cur = self.resblocks[i](e_cur)

            # for resblock in self.resblocks:
                # e_cur = resblock(e_cur+x_blocks[-1])

#########
            ## Decoder
            for i, decoder in enumerate(self.decoders):
                e_cur, state = decoder(self.apply_skip_connection(e_cur, e_blocks[self.num_encoders - i - 1]), prev_states_decoder[i])
                prev_states_decoder[i] = state

            # tail
            out = self.pred(self.apply_skip_connection(e_cur, head))
            out_l.append(out)
        
        return torch.stack(out_l, dim=1) # b,t,c,h,w


if __name__ == '__main__':
    import time

    model = MambaMotionBidirectionalNetwork(img_chn=26, ev_chn=2, num_encoders=3)
    device = 'cuda'
    x = torch.rand(1, 26, 256, 256).to(device)
    event = torch.rand(1, 24, 2, 256, 256).to(device)
    model = model.to(device)

    start_time = time.time()
    result = model(x, event)
    end_time = time.time()

    inference_time = end_time - start_time
    print('Inference time:{}'.format(inference_time))
    print('Output shape:{}'.format(result.shape))
