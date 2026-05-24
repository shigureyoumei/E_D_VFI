import torch
import torch.nn as nn

from basicsr.models.archs.EAMamba.eamamba_block import EAMambaBlock


def conv3x3(in_channels, out_channels, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


class ResidualBlock(nn.Module):
    def __init__(self, channels, relu_slope=0.2):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            conv3x3(channels, channels),
            nn.LeakyReLU(relu_slope, inplace=False),
            conv3x3(channels, channels),
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualStage(nn.Module):
    def __init__(self, channels, num_blocks=2, relu_slope=0.2):
        super(ResidualStage, self).__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(channels, relu_slope) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)


class RGBDeblurEncoder(nn.Module):
    """Appearance encoder for one blurry RGB frame.

    Returns features at 1x, 1/2x, and 1/4x resolution.
    """

    def __init__(self, in_channels=3, base_channels=32, num_blocks=2, relu_slope=0.2):
        super(RGBDeblurEncoder, self).__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.head = nn.Sequential(
            conv3x3(in_channels, c1),
            nn.LeakyReLU(relu_slope, inplace=False),
        )
        self.stage1 = ResidualStage(c1, num_blocks, relu_slope)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1, bias=False)
        self.stage2 = ResidualStage(c2, num_blocks, relu_slope)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1, bias=False)
        self.stage3 = ResidualStage(c3, num_blocks, relu_slope)

    def forward(self, x):
        x = self.head(x)
        f1 = self.stage1(x)
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))
        return f1, f2, f3


class EventEAMambaDeblurEncoder(nn.Module):
    """Exposure-event encoder for one blurry frame.

    EAMamba is applied at each spatial scale to model motion boundaries and blur
    trajectories before fusion with RGB appearance features.
    """

    def __init__(self, in_channels, base_channels=32, relu_slope=0.2,
                 scan_type="zigzag", scan_count=4, scan_merge_method="add",
                 channel_mixer_type="Simple"):
        super(EventEAMambaDeblurEncoder, self).__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.head = nn.Sequential(
            conv3x3(in_channels, c1),
            nn.LeakyReLU(relu_slope, inplace=False),
        )
        block_kwargs = dict(
            scan_type=scan_type,
            scan_count=scan_count,
            scan_merge_method=scan_merge_method,
            channel_mixer_type=channel_mixer_type,
        )
        self.block1 = EAMambaBlock(dim=c1, **block_kwargs)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1, bias=False)
        self.block2 = EAMambaBlock(dim=c2, **block_kwargs)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1, bias=False)
        self.block3 = EAMambaBlock(dim=c3, **block_kwargs)

    def forward(self, x):
        x = self.head(x)
        f1 = self.block1(x)
        f2 = self.block2(self.down1(f1))
        f3 = self.block3(self.down2(f2))
        return f1, f2, f3


class ResidualGatedFusion(nn.Module):
    def __init__(self, channels, relu_slope=0.2):
        super(ResidualGatedFusion, self).__init__()
        self.gate = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.delta = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(relu_slope, inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, rgb_feat, event_feat):
        gate = torch.sigmoid(self.gate(event_feat))
        delta = self.delta(torch.cat([rgb_feat, event_feat], dim=1))
        return rgb_feat * (1.0 + gate) + delta


class DeblurCandidateHead(nn.Module):
    """Expand one deblur feature into m exposure-window sharp candidates."""

    def __init__(self, channels, num_candidates, relu_slope=0.2):
        super(DeblurCandidateHead, self).__init__()
        self.num_candidates = num_candidates
        self.proj = nn.Sequential(
            conv3x3(channels, channels),
            nn.LeakyReLU(relu_slope, inplace=False),
            conv3x3(channels, channels * num_candidates),
        )

    def forward(self, feat):
        # feat: [B, C, H, W] -> [B, M, C, H, W]
        b, c, h, w = feat.shape
        residual = self.proj(feat).reshape(b, self.num_candidates, c, h, w)
        return feat[:, None, :, :, :] + residual


class SharedEventGuidedDeblurBranch(nn.Module):
    """Shared-weight two-frame deblur feature candidate generator.

    Input:
        b0, b1: [B, 3, H, W]
        e0, e1: [B, C_event, H, W]
    Output:
        f0, f1: tuples of three [B, M, C_s, H_s, W_s] multi-scale sharp candidates.
    """

    def __init__(self, event_channels, rgb_channels=3, base_channels=32,
                 num_rgb_blocks=2, relu_slope=0.2, num_candidates=1):
        super(SharedEventGuidedDeblurBranch, self).__init__()
        self.num_candidates = num_candidates
        self.rgb_encoder = RGBDeblurEncoder(rgb_channels, base_channels, num_rgb_blocks, relu_slope)
        self.event_encoder = EventEAMambaDeblurEncoder(event_channels, base_channels, relu_slope)
        self.fusions = nn.ModuleList([
            ResidualGatedFusion(base_channels, relu_slope),
            ResidualGatedFusion(base_channels * 2, relu_slope),
            ResidualGatedFusion(base_channels * 4, relu_slope),
        ])
        self.candidate_heads = nn.ModuleList([
            DeblurCandidateHead(base_channels, num_candidates, relu_slope),
            DeblurCandidateHead(base_channels * 2, num_candidates, relu_slope),
            DeblurCandidateHead(base_channels * 4, num_candidates, relu_slope),
        ])

    def forward_single(self, blur, exposure_event):
        rgb_feats = self.rgb_encoder(blur)
        event_feats = self.event_encoder(exposure_event)
        fused_feats = [
            fusion(rgb_feat, event_feat)
            for fusion, rgb_feat, event_feat in zip(self.fusions, rgb_feats, event_feats)
        ]
        return tuple(head(feat) for head, feat in zip(self.candidate_heads, fused_feats))

    def forward(self, b0, b1, e_exp0, e_exp1):
        f0_sharp_candidate = self.forward_single(b0, e_exp0)
        f1_sharp_candidate = self.forward_single(b1, e_exp1)
        return f0_sharp_candidate, f1_sharp_candidate
