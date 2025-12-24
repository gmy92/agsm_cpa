import torch
import torch.nn as nn
import torch.nn.functional as F

from models.risurconv_utils_robust import RISurConvSetAbstractionRobust


class get_model(nn.Module):
    def __init__(
        self,
        num_class,
        n=1,
        normal_channel=True,
        # ===== for ablation / sensitivity =====
        gating_layers=2,          # 0(no gating) / 2(early-only) / 5(all layers)
        ema_decay=0.99,
        gate_alpha=0.5,
        use_sigmoid_gate=False,
    ):
        super(get_model, self).__init__()
        self.normal_channel = normal_channel

        # -------- gating flags per layer (for ablation) --------
        # layer ids: sc0, sc1, sc2, sc3, sc4  -> 0..4
        g = [False] * 5
        for i in range(min(int(gating_layers), 5)):
            g[i] = True

        # ----- 5 层 Robust RISurConv -----
        # 512, 256, 128, 64, GroupAll

        self.sc0 = RISurConvSetAbstractionRobust(
            npoint=512 * n, radius=0.12, nsample=8,
            in_channel=0, out_channel=32,
            group_all=False, use_nr_gating=g[0],
            ema_decay=ema_decay, gate_alpha=gate_alpha, use_sigmoid_gate=use_sigmoid_gate
        )

        self.sc1 = RISurConvSetAbstractionRobust(
            npoint=256 * n, radius=0.16, nsample=16,
            in_channel=32, out_channel=64,
            group_all=False, use_nr_gating=g[1],
            ema_decay=ema_decay, gate_alpha=gate_alpha, use_sigmoid_gate=use_sigmoid_gate
        )

        self.sc2 = RISurConvSetAbstractionRobust(
            npoint=128 * n, radius=0.24, nsample=32,
            in_channel=64, out_channel=128,
            group_all=False, use_nr_gating=g[2],
            ema_decay=ema_decay, gate_alpha=gate_alpha, use_sigmoid_gate=use_sigmoid_gate
        )

        self.sc3 = RISurConvSetAbstractionRobust(
            npoint=64 * n, radius=0.48, nsample=64,
            in_channel=128, out_channel=256,
            group_all=False, use_nr_gating=g[3],
            ema_decay=ema_decay, gate_alpha=gate_alpha, use_sigmoid_gate=use_sigmoid_gate
        )

        self.sc4 = RISurConvSetAbstractionRobust(
            npoint=None, radius=None, nsample=None,
            in_channel=256, out_channel=512,
            group_all=True, use_nr_gating=g[4],
            ema_decay=ema_decay, gate_alpha=gate_alpha, use_sigmoid_gate=use_sigmoid_gate
        )

        # ----- Transformer：修复 batch_first -----
        # 你原来传的是 [B, S, C]，但 TransformerEncoderLayer 默认 batch_first=False，会错
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dropout=0.05,
            
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        # ----- 分类头 -----
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(128, num_class)

        # 可选：把 gate 保存出来便于可视化/调试
        self.last_gates = {}

    def forward(self, xyz):
        """
        xyz: [B, N, C]  (ScanObjectNN: C = 6, xyz + normal)
        """
        B, N, C = xyz.shape

        if self.normal_channel and C > 3:
            norm = xyz[:, :, 3:]
            xyz = xyz[:, :, :3]
        else:
            norm = None

        device = xyz.device
        if norm is not None:
            norm = norm.to(device)

        # ===== 5 层 Robust 特征提取 =====
        # ✅ 注意：Robust SA 层现在返回 4 个值：(..., gate_weights)
        l0_xyz, l0_norm, l0_points, g0 = self.sc0(xyz, norm, None)
        l1_xyz, l1_norm, l1_points, g1 = self.sc1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points, g2 = self.sc2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points, g3 = self.sc3(l2_xyz, l2_norm, l2_points)
        l4_xyz, l4_norm, l4_points, g4 = self.sc4(l3_xyz, l3_norm, l3_points)  # l4_points: [B, 512, 1]

        # 记录 gate（可选，用于画 heatmap/统计）
        self.last_gates = {"g0": g0, "g1": g1, "g2": g2, "g3": g3, "g4": g4}

        # Transformer: 先变成 [B, S, C]
        x = l4_points.permute(0, 2, 1) # [B, 1, 512]
        x = self.transformer_encoder(x)              # [B, 1, 512]

        # Global pooling (S = 1)
        global_x = x[:, 0, :]                        # [B, 512]

        x = self.drop1(F.relu(self.bn1(self.fc1(global_x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=-1)

        # 你原来 return x, l4_points，这里保持不变
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        return F.nll_loss(pred, target)
