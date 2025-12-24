"""
Robust RISurConv utilities with lightweight noise-robust gating.

- RISurConvSetAbstractionRobust:
    Based on original RISurConvSetAbstraction, with lightweight noise-robust gating
    applied only to the first 1-2 layers to suppress noise at early stages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Reuse original utilities
from models.risurconv_utils import (
    sample_and_group,
    sample_and_group_all,
    SA_Layer_2d,
    SA_Layer,
    index_points,
    square_distance,
    knn_point
)


class RISurConvSetAbstractionRobust(nn.Module):
    def __init__(
        self,
        npoint,
        radius,
        nsample,
        in_channel,
        out_channel,
        group_all,
        use_nr_gating=False,
        ema_decay=0.99,
        gate_alpha=0.5,
        gate_eps=1e-6,
        use_sigmoid_gate=False,
    ):
        super(RISurConvSetAbstractionRobust, self).__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.use_nr_gating = use_nr_gating

        # --- Gating hyperparams (for ablation / sensitivity) ---
        self.ema_decay = float(ema_decay)
        self.gate_alpha = float(gate_alpha)
        self.gate_eps = float(gate_eps)
        self.use_sigmoid_gate = bool(use_sigmoid_gate)

        # EMA buffer for variance normalization (paper-consistent)
        # scalar buffer; will move with device and be saved in ckpt
        self.register_buffer("ema_var", torch.zeros(1))

        # Same as original: RISP → 32 → 64
        raw_in_channel = [14, 32]
        raw_out_channel = [32, 64]

        self.embedding = nn.Sequential(
            nn.Conv2d(raw_in_channel[0], raw_out_channel[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(raw_out_channel[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(raw_out_channel[0], raw_out_channel[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(raw_out_channel[1]),
            nn.ReLU(inplace=True),
        )

        # First self-attention (within local patch)
        self.self_attention_0 = SA_Layer_2d(raw_out_channel[1])

        # Optional sigmoid gate parameters (only if you want sigmoid form)
        if self.use_nr_gating and self.use_sigmoid_gate:
            # IMPORTANT: sign will be negative in forward to make var↑ -> gate↓
            self.gate_a = nn.Parameter(torch.tensor(1.0))
            self.gate_b = nn.Parameter(torch.tensor(0.0))

        # risurconv: concatenate RISP features and previous layer features then convolve
        self.risurconv = nn.Sequential(
            nn.Conv2d(raw_out_channel[1] + in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        # Second self-attention (after max-pooling)
        self.self_attention_1 = SA_Layer(out_channel)

        # For logging/debug/return
        self._last_gate_weights = None

    def farthest_point_sample_cpu(self, xyz, npoint):
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids

    def sample_and_group_cpu(self, npoint, radius, nsample, xyz, norm):
        B, N, C = xyz.shape
        xyz = xyz.contiguous()

        fps_idx = self.farthest_point_sample_cpu(xyz, npoint)
        new_xyz = index_points(xyz, fps_idx)
        new_norm = index_points(norm, fps_idx) if norm is not None else None

        idx = knn_point(nsample, xyz, new_xyz.contiguous())

        from models.risurconv_utils import RISP_features
        ri_feat, idx_ordered = RISP_features(xyz, norm, new_xyz, new_norm, idx)
        idx = idx_ordered

        return new_xyz, ri_feat, new_norm, idx

    @torch.no_grad()
    def _update_ema(self, cur_var_scalar: torch.Tensor):
        # cur_var_scalar: scalar tensor on correct device
        self.ema_var.mul_(self.ema_decay).add_(cur_var_scalar * (1.0 - self.ema_decay))

    def forward(self, xyz, norm, points):
        """
        Input:
            xyz: [B, N, 3]
            norm: [B, N, 3] or None
            points: [B, C, N] or None
        Return:
            new_xyz: [B, S, 3]
            new_norm: [B, S, 3]
            risur_feat: [B, out_channel, S]
            gate_weights: [B, S] or None
        """
        original_device = xyz.device
        use_cuda = original_device.type == 'cuda' and torch.cuda.is_available()

        if norm is not None:
            norm = norm.to(original_device)
        if points is not None:
            points = points.permute(0, 2, 1).to(original_device)  # [B, N, C']

        B, N, _ = xyz.shape

        # Sampling + grouping
        if self.group_all:
            new_xyz, ri_feat, new_norm, idx = sample_and_group_all(xyz, norm)
            # idx is usually None in group_all path
        
            # ✅ FIX: original sample_and_group_all returns new_xyz=None by design
            if new_xyz is None:
                # Use centroid as "new_xyz" so later gating / centered computation won't crash
                new_xyz = xyz.mean(dim=1, keepdim=True)   # [B, 1, 3]
        else:
            if not use_cuda:
                new_xyz, ri_feat, new_norm, idx = self.sample_and_group_cpu(
                    self.npoint, self.radius, self.nsample, xyz, norm
                )
            else:
                new_xyz, ri_feat, new_norm, idx = sample_and_group(
                    self.npoint, self.radius, self.nsample, xyz, norm
                )

        # ri_feat: [B, S, K, 14]
        # embed: [B, 14, K, S] -> [B, 64, K, S]
        ri_feat = ri_feat.permute(0, 3, 2, 1).contiguous()
        ri_feat = self.embedding(ri_feat)
        ri_feat = self.self_attention_0(ri_feat)

        # ---------- Noise-robust gating (paper-consistent, geometry-based) ----------
        if self.use_nr_gating:
            # Compute local geometric variance from xyz neighborhoods (NOT from embedded features)
            # local_var: [B, S]
            if idx is not None:
                # idx: [B, S, K]
                grouped_xyz = index_points(xyz, idx)                    # [B, S, K, 3]
                centered = grouped_xyz - new_xyz.unsqueeze(2)           # [B, S, K, 3]
                # variance across K and 3 dims
                local_var = centered.var(dim=2).mean(dim=-1)            # [B, S]
            else:
                # group_all case: one patch per batch
                # new_xyz: [B, 1, 3], xyz: [B, N, 3]
                centered = xyz - xyz.mean(dim=1, keepdim=True)          # [B, N, 3]
                local_var = centered.var(dim=1).mean(dim=-1)            # [B]
                local_var = local_var.view(B, 1)                        # [B, 1]

            # EMA normalize (scalar EMA of batch mean variance)
            cur_var = local_var.mean()                                  # scalar
            if self.training:
                self._update_ema(cur_var.detach())
            normed_var = local_var / (self.ema_var + self.gate_eps)     # [B, S]

            # Map to reliability weight in (0,1), with correct direction: var↑ -> gate↓
            if self.use_sigmoid_gate:
                # sigmoid(-a*var + b)
                gate_w = torch.sigmoid(-self.gate_a * normed_var + self.gate_b)  # [B, S]
            else:
                # paper-friendly: 1/(1+var)
                gate_w = 1.0 / (1.0 + normed_var)
                gate_w = gate_w.clamp(0.0, 1.0)

            # store for logging/return
            self._last_gate_weights = gate_w.detach()

            # residual-style gating
            alpha = self.gate_alpha
            gate = gate_w.unsqueeze(1).unsqueeze(2)                     # [B, 1, 1, S]
            ri_feat = (1.0 - alpha) * ri_feat + alpha * gate * ri_feat  # [B, 64, K, S]
        else:
            self._last_gate_weights = None

        # ---------- Concatenate previous layer features ----------
        if points is not None:
            if idx is not None:
                grouped_points = index_points(points, idx)              # [B, S, K, C']
            else:
                grouped_points = points.view(B, 1, N, -1)               # [B, 1, N, C'] (group_all)

            grouped_points = grouped_points.permute(0, 3, 2, 1)          # [B, C', K, S]
            new_points = torch.cat([ri_feat, grouped_points], dim=1)
        else:
            new_points = ri_feat

        # Local conv
        new_points = self.risurconv(new_points)                         # [B, out_channel, K, S]

        # Max pooling over K
        risur_feat = torch.max(new_points, 2)[0]                         # [B, out_channel, S]

        # Global self-attention
        risur_feat = self.self_attention_1(risur_feat)

        return new_xyz, new_norm, risur_feat, self._last_gate_weights
