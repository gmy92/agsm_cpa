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
    def __init__(self, npoint, radius, nsample, in_channel, out_channel, group_all, use_nr_gating=False):
        super(RISurConvSetAbstractionRobust, self).__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.use_nr_gating = use_nr_gating  # Whether to use noise-robust gating

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

        # *** Lightweight noise-robust gating (applied only to early layers) ***
        if self.use_nr_gating:
            # Learnable scalar parameters for gating
            self.gate_a = nn.Parameter(torch.tensor(1.0))  # Scale parameter
            self.gate_b = nn.Parameter(torch.tensor(0.0))  # Bias parameter

        # risurconv: concatenate RISP features and previous layer features then convolve
        self.risurconv = nn.Sequential(
            nn.Conv2d(raw_out_channel[1] + in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        # Second self-attention (after max-pooling)
        self.self_attention_1 = SA_Layer(out_channel)

    def farthest_point_sample_cpu(self, xyz, npoint):
        """
        CPU implementation of farthest point sampling
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
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
        """
        CPU-compatible version of sample_and_group
        """
        B, N, C = xyz.shape
        xyz = xyz.contiguous()
        
        # Use farthest point sampling on CPU
        fps_idx = self.farthest_point_sample_cpu(xyz, npoint)
        new_xyz = index_points(xyz, fps_idx)
        new_norm = index_points(norm, fps_idx) if norm is not None else None
        
        # Use KNN for grouping
        idx = knn_point(nsample, xyz, new_xyz.contiguous())
        
        # Get RISP features
        from models.risurconv_utils import RISP_features
        ri_feat, idx_ordered = RISP_features(xyz, norm, new_xyz, new_norm, idx)
        idx = idx_ordered
        
        return new_xyz, ri_feat, new_norm, idx

    def forward(self, xyz, norm, points):
        """
        Input:
            xyz: [B, N, 3]
            norm: [B, N, 3] or None
            points: [B, C, N] or None (previous layer features)
        Return:
            new_xyz: [B, S, 3]
            new_norm: [B, S, 3]
            risur_feat: [B, out_channel, S]
        """
        # Store original device
        original_device = xyz.device
        
        # For CPU tensors, we need to handle pointops operations differently
        use_cuda = original_device.type == 'cuda' and torch.cuda.is_available()
        
        # Ensure all inputs are on the same device
        if norm is not None:
            norm = norm.to(original_device)
        if points is not None:  # [B, C, N] -> [B, N, C]
            points = points.permute(0, 2, 1).to(original_device)

        B, N, C = xyz.shape

        # Sampling + grouping
        if self.group_all:
            new_xyz, ri_feat, new_norm, idx = sample_and_group_all(xyz, norm)
        else:
            # Use CPU-compatible implementation when on CPU
            if not use_cuda:
                new_xyz, ri_feat, new_norm, idx = self.sample_and_group_cpu(
                    self.npoint, self.radius, self.nsample, xyz, norm
                )
            else:
                # Use original implementation for CUDA
                new_xyz, ri_feat, new_norm, idx = sample_and_group(
                    self.npoint, self.radius, self.nsample, xyz, norm
                )

        # ri_feat: [B, S, K, F], where F=14

        # Embed to 64-dimensional channel space
        # -> [B, F, K, S] -> embedding -> [B, 64, K, S]
        ri_feat = ri_feat.permute(0, 3, 2, 1).contiguous()
        ri_feat = self.embedding(ri_feat)
        ri_feat = self.self_attention_0(ri_feat)   # Still [B, 64, K, S]

        # ---------- Lightweight noise-robust gating (only for early layers) ----------
        if self.use_nr_gating:
            # 1) Compute local variance as "noise score"
            # Using local coordinate variance (first 3 channels)
            local_coords = ri_feat[:, :3, :, :]  # [B, 3, K, S]
            local_mean = local_coords.mean(dim=2, keepdim=True)  # [B, 3, 1, S]
            local_var = (local_coords - local_mean).pow(2).mean(dim=2)  # [B, 3, S]
            local_var = local_var.mean(dim=1)  # [B, S] - average across coordinate dimensions

            # 2) Map to reliability weight in (0,1)
            gate = torch.sigmoid(self.gate_a * local_var + self.gate_b)  # [B, S]
            gate = gate.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S] for broadcasting

            # 3) Residual-style gating to avoid completely shutting down features
            alpha = 0.5  # Hyperparameter, 0~1
            # Apply gating to features before risurconv
            ri_feat_gated = (1 - alpha) * ri_feat + alpha * gate * ri_feat  # [B, 64, K, S]
            ri_feat = ri_feat_gated

        # ---------- Concatenate previous layer features ----------
        if points is not None:
            if idx is not None:
                # idx: [B, S, K]
                grouped_points = index_points(points, idx)  # [B, S, K, C']
            else:
                grouped_points = points.view(B, 1, N, -1)   # group_all case

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, C', K, S]
            new_points = torch.cat([ri_feat, grouped_points], dim=1)
        else:
            new_points = ri_feat

        # Local RBF + convolution
        new_points = self.risurconv(new_points)  # [B, out_channel, K, S]

        # Max pooling over K
        risur_feat = torch.max(new_points, 2)[0]  # [B, out_channel, S]

        # Another global self-attention
        risur_feat = self.self_attention_1(risur_feat)

        return new_xyz, new_norm, risur_feat,self._last_gate_weights