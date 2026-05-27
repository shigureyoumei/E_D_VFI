import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.arch_util import flow_warp

try:
    from basicsr.models.archs.mamba.EAMamba.eamamba_block import EAMambaBlock
except Exception:
    EAMambaBlock = None


def sinusoidal_embedding(time_ids, dim):
    """Build sinusoidal embeddings for normalized target/event times.

    Args:
        time_ids: [T]
        dim: embedding dimension
    Returns:
        [T, dim]
    """
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
        emb = F.pad(emb, (0, 1))
    return emb


def polynomial_time_basis(tau, num_basis):
    """Polynomial coefficients phi(tau) = [tau, tau^2, ...].

    Args:
        tau: [T], values in (0, 1)
        num_basis: K
    Returns:
        [T, K]
    """
    return torch.stack([tau.pow(k + 1) for k in range(num_basis)], dim=-1)


def get_polynomial_weights(t_values, device, dtype):
    """Return continuous motion and mask weights for the four shared bases."""
    t = torch.as_tensor(t_values, device=device, dtype=dtype).flatten()
    one_minus_t = 1.0 - t
    shared = t * one_minus_t
    w0 = torch.stack((t, t.pow(2), t.pow(3), shared), dim=-1)
    w1 = torch.stack((one_minus_t, one_minus_t.pow(2), one_minus_t.pow(3), shared), dim=-1)
    wm = torch.stack((torch.ones_like(t), t, one_minus_t, shared), dim=-1)
    return w0, w1, wm


def resize_flow(flow, size):
    """Resize flow and scale its pixel displacement magnitude.

    Args:
        flow: [B, 2, H, W] in source-resolution pixels
        size: target (H_out, W_out)
    Returns:
        [B, 2, H_out, W_out] in target-resolution pixels
    """
    h, w = flow.shape[-2:]
    out_h, out_w = size
    if (h, w) == (out_h, out_w):
        return flow

    flow = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
    scale_x = out_w / w
    scale_y = out_h / h
    scale = flow.new_tensor((scale_x, scale_y)).reshape(1, 2, 1, 1)
    return flow * scale


class MotionResidualBlock(nn.Module):
    """Small residual block used by the stable CNN motion-basis encoder."""

    def __init__(self, channels):
        super(MotionResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class EventGuidedMotionBasis(nn.Module):
    """Generate continuous-time interpolation features from shared motion bases.

    Instead of predicting an independent optical flow for every target time,
    this module predicts shared dense motion bases once and composes
    time-specific flow and mask fields with polynomial time weights.
    """

    def __init__(self, in_channels, hidden_channels=64, num_basis=4,
                 use_eamamba=False, refine=True, feature_channels=None,
                 event_channels=0):
        super(EventGuidedMotionBasis, self).__init__()
        if num_basis != 4:
            raise NotImplementedError('EventGuidedMotionBasis currently supports num_basis=4 only.')
        self.num_basis = num_basis
        self.refine = refine
        self.feature_channels = feature_channels if feature_channels is not None else in_channels
        self.event_channels = event_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=False),
            MotionResidualBlock(hidden_channels),
            MotionResidualBlock(hidden_channels),
        )
        if use_eamamba:
            if EAMambaBlock is None:
                raise ImportError('EAMambaBlock is unavailable. Set use_eamamba=False.')
            self.eamamba = EAMambaBlock(dim=hidden_channels)
        else:
            self.eamamba = nn.Identity()

        self.basis0_head = nn.Conv2d(hidden_channels, num_basis * 2, kernel_size=3, stride=1, padding=1)
        self.basis1_head = nn.Conv2d(hidden_channels, num_basis * 2, kernel_size=3, stride=1, padding=1)
        self.mask_head = nn.Conv2d(hidden_channels, num_basis, kernel_size=3, stride=1, padding=1)
        for head in (self.basis0_head, self.basis1_head, self.mask_head):
            nn.init.normal_(head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(head.bias)

        if refine:
            refine_in_channels = self.feature_channels + event_channels + 1
            self.refinement_conv = nn.Sequential(
                nn.Conv2d(refine_in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
                nn.SiLU(inplace=False),
                nn.Conv2d(hidden_channels, self.feature_channels, kernel_size=3, stride=1, padding=1),
            )
            nn.init.normal_(self.refinement_conv[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.refinement_conv[-1].bias)

    def _warp_multitime(self, feat, flows):
        b, t, _, h, w = flows.shape
        feat_rep = feat[:, None].expand(-1, t, -1, -1, -1).reshape(b * t, feat.size(1), h, w)
        flow_flat = flows.reshape(b * t, 2, h, w).permute(0, 2, 3, 1)
        warped = flow_warp(feat_rep, flow_flat, padding_mode='border')
        return warped.reshape(b, t, feat.size(1), h, w)

    def forward(self, feat_fused, feat0, feat1, t_values, event_feats_t=None):
        if feat0.shape != feat1.shape:
            raise ValueError('feat0 and feat1 must have identical shapes.')
        if feat0.size(1) != self.feature_channels:
            raise ValueError(
                f'Expected endpoint features with {self.feature_channels} channels, got {feat0.size(1)}.'
            )
        t_values = torch.as_tensor(t_values, device=feat_fused.device, dtype=feat_fused.dtype).flatten()
        b, _, h, w = feat_fused.shape
        t = t_values.numel()
        if event_feats_t is not None and event_feats_t.shape[:2] != (b, t):
            raise ValueError('event_feats_t must have shape [B, T, C_event, H, W].')

        encoded = self.eamamba(self.encoder(feat_fused))
        basis_0 = self.basis0_head(encoded).reshape(b, self.num_basis, 2, h, w)
        basis_1 = self.basis1_head(encoded).reshape(b, self.num_basis, 2, h, w)
        mask_basis = self.mask_head(encoded).reshape(b, self.num_basis, 1, h, w)
        w0, w1, wm = get_polynomial_weights(t_values, feat_fused.device, feat_fused.dtype)
        flows_0_to_t = torch.einsum('tk,bkchw->btchw', w0, basis_0)
        flows_1_to_t = torch.einsum('tk,bkchw->btchw', w1, basis_1)
        masks = torch.sigmoid(torch.einsum('tk,bkchw->btchw', wm, mask_basis))

        warped_0 = self._warp_multitime(feat0, flows_0_to_t)
        warped_1 = self._warp_multitime(feat1, flows_1_to_t)
        interp_feats = masks * warped_0 + (1.0 - masks) * warped_1

        if self.refine:
            if event_feats_t is None:
                event_feats_t = feat_fused.new_zeros(b, t, self.event_channels, h, w)
            elif event_feats_t.size(2) != self.event_channels:
                raise ValueError(
                    f'Expected {self.event_channels} event channels, got {event_feats_t.size(2)}.'
                )
            time_maps = t_values.reshape(1, t, 1, 1, 1).expand(b, -1, -1, h, w)
            refine_input = torch.cat((interp_feats, event_feats_t, time_maps), dim=2)
            residual = self.refinement_conv(refine_input.reshape(
                b * t, refine_input.size(2), h, w))
            interp_feats = interp_feats + residual.reshape(b, t, self.feature_channels, h, w)

        debug = {
            'basis_0': basis_0,
            'basis_1': basis_1,
            'mask_basis': mask_basis,
            'flows_0_to_t': flows_0_to_t,
            'flows_1_to_t': flows_1_to_t,
            'masks': masks,
        }
        return interp_feats, debug


def warp_feature(feat, flow):
    """Differentiable backward warping with pixel-unit flow.

    Args:
        feat: [B, C, H, W]
        flow: [B, 2, H, W], flow[:,0] is x displacement, flow[:,1] is y displacement
    Returns:
        warped: [B, C, H, W]
    """
    b, _, h, w = feat.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=feat.device, dtype=feat.dtype),
        torch.arange(w, device=feat.device, dtype=feat.dtype),
        indexing="ij",
    )
    grid = torch.stack((x, y), dim=0).unsqueeze(0).expand(b, -1, -1, -1)
    sample_grid = grid + flow
    sample_x = 2.0 * sample_grid[:, 0] / max(w - 1, 1) - 1.0
    sample_y = 2.0 * sample_grid[:, 1] / max(h - 1, 1) - 1.0
    sample_grid = torch.stack((sample_x, sample_y), dim=-1)
    return F.grid_sample(feat, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)


class TemporalMambaBlock(nn.Module):
    """Bidirectional temporal EAMamba over per-pixel event sequences.

    Input/output shape: [B, T, C, H, W].
    The implementation treats each spatial location as a length-T sequence.
    """

    def __init__(self, channels, use_eamamba=True):
        super(TemporalMambaBlock, self).__init__()
        self.channels = channels
        self.time_mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.use_eamamba = use_eamamba
        if use_eamamba:
            if EAMambaBlock is None:
                raise ImportError("EAMambaBlock is unavailable. Set use_eamamba=False for the GRU fallback.")
            self.forward_mamba = EAMambaBlock(dim=channels)
            self.backward_mamba = EAMambaBlock(dim=channels)
        else:
            self.forward_mamba = nn.GRU(channels, channels, batch_first=True)
            self.backward_mamba = nn.GRU(channels, channels, batch_first=True)
        self.proj = nn.Linear(channels * 2, channels)

    def _run_sequence_block(self, block, x_seq):
        # x_seq: [N, T, C]
        if self.use_eamamba:
            x_4d = x_seq.transpose(1, 2).unsqueeze(-1)  # [N, C, T, 1]
            return block(x_4d).squeeze(-1).transpose(1, 2)
        y, _ = block(x_seq)
        return y

    def forward(self, x):
        b, t, c, h, w = x.shape
        x_seq = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)

        time_ids = torch.linspace(0, 1, t, device=x.device, dtype=x.dtype)
        time_emb = self.time_mlp(sinusoidal_embedding(time_ids, c))
        x_seq = x_seq + time_emb[None, :, :]

        y_f = self._run_sequence_block(self.forward_mamba, x_seq)
        y_b = torch.flip(
            self._run_sequence_block(self.backward_mamba, torch.flip(x_seq, dims=[1])),
            dims=[1],
        )
        y = self.proj(torch.cat([y_f, y_b], dim=-1))
        return y.reshape(b, h, w, t, c).permute(0, 3, 4, 1, 2)


class TemporalEventPyramidEncoder(nn.Module):
    """Temporal event encoder that returns a three-scale event feature pyramid."""

    def __init__(self, event_channels, base_channels=32, use_eamamba=True, relu_slope=0.2):
        super(TemporalEventPyramidEncoder, self).__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.head = nn.Sequential(
            nn.Conv2d(event_channels, c1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(relu_slope, inplace=False),
        )
        self.temporal1 = TemporalMambaBlock(c1, use_eamamba=use_eamamba)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1, bias=False)
        self.temporal2 = TemporalMambaBlock(c2, use_eamamba=use_eamamba)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1, bias=False)
        self.temporal3 = TemporalMambaBlock(c3, use_eamamba=use_eamamba)

    def _apply_2d(self, module, x):
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = module(x)
        _, c_out, h_out, w_out = x.shape
        return x.reshape(b, t, c_out, h_out, w_out)

    def forward(self, event_seq):
        # event_seq: [B, T, C_event, H, W]
        f1 = self.temporal1(self._apply_2d(self.head, event_seq))
        f2 = self.temporal2(self._apply_2d(self.down1, f1))
        f3 = self.temporal3(self._apply_2d(self.down2, f2))
        return [f1, f2, f3]


class MotionBasisFlowPredictor(nn.Module):
    """Predict K forward and backward basis flow fields at the deepest scale."""

    def __init__(self, in_channels, num_basis=3, hidden_channels=None):
        super(MotionBasisFlowPredictor, self).__init__()
        hidden_channels = hidden_channels or in_channels
        self.num_basis = num_basis
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(hidden_channels, num_basis * 4, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, event_feat):
        # event_feat: [B, C, H, W]
        b, _, h, w = event_feat.shape
        flow = self.net(event_feat).reshape(b, self.num_basis, 4, h, w)
        basis_flows_0 = flow[:, :, 0:2]
        basis_flows_1 = flow[:, :, 2:4]
        return basis_flows_0, basis_flows_1


class ScaleInterpolationHead(nn.Module):
    """Per-scale mask prediction and feature fusion."""

    def __init__(self, channels, event_channels, hidden_channels=None):
        super(ScaleInterpolationHead, self).__init__()
        hidden_channels = hidden_channels or channels
        in_channels = channels * 3 + event_channels + 1
        mask_channels = channels * 2 + event_channels
        self.mask = nn.Sequential(
            nn.Conv2d(mask_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, stride=1, padding=1),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, f0_warp, f1_warp, event_feat):
        # All inputs are [B*T, C, H, W] except mask output [B*T, 1, H, W].
        mask = torch.sigmoid(self.mask(torch.cat([f0_warp, f1_warp, event_feat], dim=1)))
        blended = mask * f0_warp + (1.0 - mask) * f1_warp
        interp = self.fusion(torch.cat([f0_warp, f1_warp, blended, event_feat, mask], dim=1))
        return interp, mask, blended


class BasisScaleRefinementHead(nn.Module):
    """Refine a basis-warped blend with target-time event evidence."""

    def __init__(self, channels, event_channels, hidden_channels=None):
        super(BasisScaleRefinementHead, self).__init__()
        hidden_channels = hidden_channels or channels
        self.body = nn.Sequential(
            nn.Conv2d(channels + event_channels + 1, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=False),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, stride=1, padding=1),
        )
        nn.init.normal_(self.body[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, blended, event_feat, tau):
        b, t, _, h, w = blended.shape
        time_maps = tau.to(device=blended.device, dtype=blended.dtype).reshape(1, t, 1, 1, 1)
        time_maps = time_maps.expand(b, -1, -1, h, w)
        inputs = torch.cat((blended, event_feat, time_maps), dim=2)
        residual = self.body(inputs.reshape(b * t, inputs.size(2), h, w))
        return blended + residual.reshape(b, t, blended.size(2), h, w)


class MotionBasisInterpolationBranch(nn.Module):
    """Motion-basis event-guided feature interpolation branch.

    Args:
        feature_channels: channels for [s1, s2, s3] deblur features
        event_channels: input event voxel channels
        num_basis: number of polynomial motion bases K
    """

    def __init__(self, feature_channels=(32, 64, 128), event_channels=2,
                 base_event_channels=32, num_basis=None, use_eamamba=True,
                 use_motion_basis=True, motion_basis_use_eamamba=False):
        super(MotionBasisInterpolationBranch, self).__init__()
        if len(feature_channels) != 3:
            raise ValueError("This first implementation expects exactly three feature scales.")
        self.use_motion_basis = use_motion_basis
        self.feature_channels = tuple(feature_channels)
        self.num_basis = num_basis if num_basis is not None else (4 if use_motion_basis else 3)
        self.event_encoder = TemporalEventPyramidEncoder(
            event_channels=event_channels,
            base_channels=base_event_channels,
            use_eamamba=use_eamamba,
        )
        event_pyramid_channels = (base_event_channels, base_event_channels * 2, base_event_channels * 4)
        if use_motion_basis:
            deep_channels = self.feature_channels[-1]
            self.motion_basis = EventGuidedMotionBasis(
                in_channels=deep_channels * 2 + event_pyramid_channels[-1],
                hidden_channels=64,
                num_basis=self.num_basis,
                use_eamamba=motion_basis_use_eamamba,
                refine=True,
                feature_channels=deep_channels,
                event_channels=event_pyramid_channels[-1],
            )
            self.scale_refinement_heads = nn.ModuleList([
                BasisScaleRefinementHead(feat_c, evt_c)
                for feat_c, evt_c in zip(self.feature_channels[:-1], event_pyramid_channels[:-1])
            ])
        else:
            self.flow_predictor = MotionBasisFlowPredictor(event_pyramid_channels[-1], num_basis=self.num_basis)
            self.scale_heads = nn.ModuleList([
                ScaleInterpolationHead(feat_c, evt_c)
                for feat_c, evt_c in zip(self.feature_channels, event_pyramid_channels)
            ])

    def _compose_flows(self, basis_flows, tau):
        # basis_flows: [B, K, 2, H, W], tau: [T], returns [B, T, 2, H, W]
        coeffs = polynomial_time_basis(tau.to(device=basis_flows.device, dtype=basis_flows.dtype), self.num_basis)
        return torch.einsum("tk,bkchw->btchw", coeffs, basis_flows)

    def _warp_multitime(self, feat, flows):
        # feat: [B, C, H, W], flows: [B, T, 2, H, W], returns [B, T, C, H, W]
        b, t, _, h, w = flows.shape
        feat_rep = feat[:, None].expand(-1, t, -1, -1, -1).reshape(b * t, feat.shape[1], h, w)
        flows = flows.reshape(b * t, 2, h, w)
        warped = warp_feature(feat_rep, flows)
        return warped.reshape(b, t, feat.shape[1], h, w)

    def _forward_motion_basis(self, f0_deblur, f1_deblur, event_feats, tau, return_debug):
        b, t = event_feats[-1].shape[:2]
        event_global = event_feats[-1].mean(dim=1)
        feat_fused = torch.cat((f0_deblur[-1], f1_deblur[-1], event_global), dim=1)
        deep_interp, basis_debug = self.motion_basis(
            feat_fused, f0_deblur[-1], f1_deblur[-1], tau, event_feats_t=event_feats[-1])

        interp_features = []
        masks = []
        flows_0 = []
        flows_1 = []
        for f0_s, f1_s, event_s, refine_head in zip(
                f0_deblur[:-1], f1_deblur[:-1], event_feats[:-1], self.scale_refinement_heads):
            h_s, w_s = f0_s.shape[-2:]
            flow0_s = resize_flow(
                basis_debug['flows_0_to_t'].reshape(b * t, 2, *basis_debug['flows_0_to_t'].shape[-2:]),
                (h_s, w_s)).reshape(b, t, 2, h_s, w_s)
            flow1_s = resize_flow(
                basis_debug['flows_1_to_t'].reshape(b * t, 2, *basis_debug['flows_1_to_t'].shape[-2:]),
                (h_s, w_s)).reshape(b, t, 2, h_s, w_s)
            mask_s = F.interpolate(
                basis_debug['masks'].reshape(b * t, 1, *basis_debug['masks'].shape[-2:]),
                size=(h_s, w_s), mode='bilinear', align_corners=True).reshape(b, t, 1, h_s, w_s)
            f0_warp = self.motion_basis._warp_multitime(f0_s, flow0_s)
            f1_warp = self.motion_basis._warp_multitime(f1_s, flow1_s)
            blended = mask_s * f0_warp + (1.0 - mask_s) * f1_warp
            interp_features.append(refine_head(blended, event_s, tau))
            if return_debug:
                masks.append(mask_s.detach())
                flows_0.append(flow0_s.detach())
                flows_1.append(flow1_s.detach())

        interp_features.append(deep_interp)
        if not return_debug:
            return interp_features, None

        masks.append(basis_debug['masks'].detach())
        flows_0.append(basis_debug['flows_0_to_t'].detach())
        flows_1.append(basis_debug['flows_1_to_t'].detach())
        debug = {
            'basis_0': basis_debug['basis_0'].detach(),
            'basis_1': basis_debug['basis_1'].detach(),
            'mask_basis': basis_debug['mask_basis'].detach(),
            'flows_0_to_t': flows_0,
            'flows_1_to_t': flows_1,
            'masks': masks,
            'event_features': [feat.detach() for feat in event_feats],
        }
        return interp_features, debug

    def _forward_legacy(self, f0_deblur, f1_deblur, event_feats, tau, return_debug):
        b, t = event_feats[-1].shape[:2]
        event_global = event_feats[-1].mean(dim=1)
        basis_flows_0, basis_flows_1 = self.flow_predictor(event_global)
        low_flows_0 = self._compose_flows(basis_flows_0, tau)
        low_flows_1 = self._compose_flows(basis_flows_1, 1.0 - tau)

        interp_features = []
        if return_debug:
            masks = []
            flows_0 = []
            flows_1 = []
            warped_0 = []
            warped_1 = []
        for f0_s, f1_s, event_s, head in zip(f0_deblur, f1_deblur, event_feats, self.scale_heads):
            _, c_s, h_s, w_s = f0_s.shape
            flow0_s = resize_flow(low_flows_0.reshape(b * t, 2, *low_flows_0.shape[-2:]), (h_s, w_s))
            flow1_s = resize_flow(low_flows_1.reshape(b * t, 2, *low_flows_1.shape[-2:]), (h_s, w_s))
            flow0_s = flow0_s.reshape(b, t, 2, h_s, w_s)
            flow1_s = flow1_s.reshape(b, t, 2, h_s, w_s)
            f0_warp = self._warp_multitime(f0_s, flow0_s)
            f1_warp = self._warp_multitime(f1_s, flow1_s)
            interp_flat, mask_flat, _ = head(
                f0_warp.reshape(b * t, c_s, h_s, w_s),
                f1_warp.reshape(b * t, c_s, h_s, w_s),
                event_s.reshape(b * t, event_s.size(2), h_s, w_s))
            interp_features.append(interp_flat.reshape(b, t, c_s, h_s, w_s))
            if return_debug:
                masks.append(mask_flat.detach().reshape(b, t, 1, h_s, w_s))
                flows_0.append(flow0_s.detach())
                flows_1.append(flow1_s.detach())
                warped_0.append(f0_warp.detach())
                warped_1.append(f1_warp.detach())
        if not return_debug:
            return interp_features, None
        return interp_features, {
            'basis_flows_0': basis_flows_0.detach(),
            'basis_flows_1': basis_flows_1.detach(),
            'flows_0_to_t': flows_0,
            'flows_1_to_t': flows_1,
            'masks': masks,
            'warped_0': warped_0,
            'warped_1': warped_1,
            'event_features': [feat.detach() for feat in event_feats],
        }

    def forward(self, f0_deblur, f1_deblur, e_between, tau, return_debug=False):
        """Interpolate multi-scale features.

        Args:
            f0_deblur/f1_deblur: lists or tuples of three tensors, each [B, C_s, H_s, W_s]
            e_between: [B, T, C_event, H, W]
            tau: [T], target times in (0, 1)

        Returns:
            interp_features: list of three [B, T, C_s, H_s, W_s] tensors
            debug: dict with basis flows, generated flows, masks, warped features
        """
        if len(f0_deblur) != 3 or len(f1_deblur) != 3:
            raise ValueError("f0_deblur and f1_deblur must contain three scales.")
        if tau.dim() != 1:
            raise ValueError("tau must be a 1D tensor with shape [T].")
        b, t, _, _, _ = e_between.shape
        if tau.numel() != t:
            raise ValueError(f"tau length ({tau.numel()}) must match event sequence T ({t}).")

        event_feats = self.event_encoder(e_between)  # three [B, T, C_s, H_s, W_s] tensors
        if self.use_motion_basis:
            return self._forward_motion_basis(f0_deblur, f1_deblur, event_feats, tau, return_debug)
        return self._forward_legacy(f0_deblur, f1_deblur, event_feats, tau, return_debug)


if __name__ == "__main__":
    torch.manual_seed(0)
    module = EventGuidedMotionBasis(
        in_channels=64,
        hidden_channels=64,
        num_basis=4,
        use_eamamba=False,
        refine=True,
    )
    features = torch.randn(1, 64, 64, 64)
    interp, debug = module(features, features, features, torch.tensor([0.25, 0.5, 0.75]))
    print(tuple(interp.shape))
    print(tuple(debug['flows_0_to_t'].shape), tuple(debug['masks'].shape))

    branch = MotionBasisInterpolationBranch(
        feature_channels=(8, 16, 32),
        event_channels=2,
        base_event_channels=8,
        num_basis=4,
        use_eamamba=False,
    )
    b, t, h, w = 2, 3, 64, 64
    f0 = [
        torch.randn(b, 8, h, w),
        torch.randn(b, 16, h // 2, w // 2),
        torch.randn(b, 32, h // 4, w // 4),
    ]
    f1 = [
        torch.randn(b, 8, h, w),
        torch.randn(b, 16, h // 2, w // 2),
        torch.randn(b, 32, h // 4, w // 4),
    ]
    events = torch.randn(b, t, 2, h, w)
    tau = torch.tensor([0.25, 0.5, 0.75])
    out, aux = branch(f0, f1, events, tau, return_debug=True)
    print([tuple(x.shape) for x in out])
    print([tuple(x.shape) for x in aux["masks"]])
    sum(item.mean() for item in out).backward()
    assert branch.motion_basis.basis0_head.weight.grad is not None
    assert not aux['basis_0'].requires_grad

    legacy_branch = MotionBasisInterpolationBranch(
        feature_channels=(8, 16, 32),
        event_channels=2,
        base_event_channels=8,
        use_eamamba=False,
        use_motion_basis=False,
    )
    legacy_out, _ = legacy_branch(f0, f1, events, tau)
    print([tuple(x.shape) for x in legacy_out])
