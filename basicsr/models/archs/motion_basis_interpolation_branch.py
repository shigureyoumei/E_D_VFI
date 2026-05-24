import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from basicsr.models.archs.EAMamba.eamamba_block import EAMambaBlock
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
    flow = flow.clone()
    flow[:, 0] *= scale_x
    flow[:, 1] *= scale_y
    return flow


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


class MotionBasisInterpolationBranch(nn.Module):
    """Motion-basis event-guided feature interpolation branch.

    Args:
        feature_channels: channels for [s1, s2, s3] deblur features
        event_channels: input event voxel channels
        num_basis: number of polynomial motion bases K
    """

    def __init__(self, feature_channels=(32, 64, 128), event_channels=2,
                 base_event_channels=32, num_basis=3, use_eamamba=True):
        super(MotionBasisInterpolationBranch, self).__init__()
        if len(feature_channels) != 3:
            raise ValueError("This first implementation expects exactly three feature scales.")
        self.feature_channels = tuple(feature_channels)
        self.num_basis = num_basis
        self.event_encoder = TemporalEventPyramidEncoder(
            event_channels=event_channels,
            base_channels=base_event_channels,
            use_eamamba=use_eamamba,
        )
        event_pyramid_channels = (base_event_channels, base_event_channels * 2, base_event_channels * 4)
        self.flow_predictor = MotionBasisFlowPredictor(event_pyramid_channels[-1], num_basis=num_basis)
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

    def forward(self, f0_deblur, f1_deblur, e_between, tau):
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
        event_global = event_feats[-1].mean(dim=1)   # [B, C_3, H_3, W_3]
        basis_flows_0, basis_flows_1 = self.flow_predictor(event_global)

        low_flows_0 = self._compose_flows(basis_flows_0, tau)
        low_flows_1 = self._compose_flows(basis_flows_1, 1.0 - tau)

        interp_features = []
        masks = []
        flows_0 = []
        flows_1 = []
        warped_0 = []
        warped_1 = []

        for scale_idx, (f0_s, f1_s, event_s, head) in enumerate(zip(f0_deblur, f1_deblur, event_feats, self.scale_heads)):
            _, c_s, h_s, w_s = f0_s.shape
            flow0_s = low_flows_0.reshape(b * t, 2, low_flows_0.shape[-2], low_flows_0.shape[-1])
            flow1_s = low_flows_1.reshape(b * t, 2, low_flows_1.shape[-2], low_flows_1.shape[-1])
            flow0_s = resize_flow(flow0_s, (h_s, w_s)).reshape(b, t, 2, h_s, w_s)
            flow1_s = resize_flow(flow1_s, (h_s, w_s)).reshape(b, t, 2, h_s, w_s)

            f0_warp = self._warp_multitime(f0_s, flow0_s)
            f1_warp = self._warp_multitime(f1_s, flow1_s)

            f0_flat = f0_warp.reshape(b * t, c_s, h_s, w_s)
            f1_flat = f1_warp.reshape(b * t, c_s, h_s, w_s)
            event_flat = event_s.reshape(b * t, event_s.shape[2], h_s, w_s)
            interp_flat, mask_flat, _ = head(f0_flat, f1_flat, event_flat)

            interp_features.append(interp_flat.reshape(b, t, c_s, h_s, w_s))
            masks.append(mask_flat.reshape(b, t, 1, h_s, w_s))
            flows_0.append(flow0_s)
            flows_1.append(flow1_s)
            warped_0.append(f0_warp)
            warped_1.append(f1_warp)

        debug = {
            "basis_flows_0": basis_flows_0,
            "basis_flows_1": basis_flows_1,
            "flows_0_to_t": flows_0,
            "flows_1_to_t": flows_1,
            "masks": masks,
            "warped_0": warped_0,
            "warped_1": warped_1,
            "event_features": event_feats,
        }
        return interp_features, debug


if __name__ == "__main__":
    torch.manual_seed(0)
    branch = MotionBasisInterpolationBranch(
        feature_channels=(8, 16, 32),
        event_channels=2,
        base_event_channels=8,
        num_basis=3,
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
    out, aux = branch(f0, f1, events, tau)
    print([tuple(x.shape) for x in out])
    print([tuple(x.shape) for x in aux["masks"]])
