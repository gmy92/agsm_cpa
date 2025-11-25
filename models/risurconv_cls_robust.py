import torch
import torch.nn as nn
import torch.nn.functional as F

from models.risurconv_utils_robust import RISurConvSetAbstractionRobust


class get_model(nn.Module):
    def __init__(self, num_class, n=1, normal_channel=True):
        super(get_model, self).__init__()
        self.normal_channel = normal_channel

        # ----- 5 层 Robust RISurConv -----
        # 恢复原版下采样策略: 512, 256, 128, 64, GroupAll

        # Layer 1: Robust + NR gating
        self.sc0 = RISurConvSetAbstractionRobust(
            npoint=512 * n, radius=0.12, nsample=8,
            in_channel=0, out_channel=32,
            group_all=False, use_nr_gating=True
        )

        # Layer 2: Robust + NR gating
        self.sc1 = RISurConvSetAbstractionRobust(
            npoint=256 * n, radius=0.16, nsample=16,
            in_channel=32, out_channel=64,
            group_all=False, use_nr_gating=True
        )

        # Layer 3: Robust (no NR gating)
        self.sc2 = RISurConvSetAbstractionRobust(
            npoint=128 * n, radius=0.24, nsample=32,
            in_channel=64, out_channel=128,
            group_all=False, use_nr_gating=False
        )

        # Layer 4: Robust (no NR gating)
        self.sc3 = RISurConvSetAbstractionRobust(
            npoint=64 * n, radius=0.48, nsample=64,
            in_channel=128, out_channel=256,
            group_all=False, use_nr_gating=False
        )

        # Layer 5: Robust (no NR gating), group_all
        self.sc4 = RISurConvSetAbstractionRobust(
            npoint=None, radius=None, nsample=None,
            in_channel=256, out_channel=512,
            group_all=True, use_nr_gating=False
        )

        # ----- Transformer：恢复原版设置 -----
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dropout=0.05,       # 恢复原版 dropout=0.05
            norm_first=False,   # 恢复原版 PostNorm
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=3
        )

        # ----- 分类头：恢复原版结构 -----
        self.fc1 = nn.Linear(512, 256)  # 恢复原版结构: 512->256
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(256, 128)  # 恢复原版结构: 256->128
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(128, num_class)  # 恢复原版结构: 128->num_class

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

        # 5 层 Robust 特征提取
        l0_xyz, l0_norm, l0_points = self.sc0(xyz, norm, None)
        l1_xyz, l1_norm, l1_points = self.sc1(l0_xyz, l0_norm, l0_points)
        l2_xyz, l2_norm, l2_points = self.sc2(l1_xyz, l1_norm, l1_points)
        l3_xyz, l3_norm, l3_points = self.sc3(l2_xyz, l2_norm, l2_points)
        l4_xyz, l4_norm, l4_points = self.sc4(l3_xyz, l3_norm, l3_points)  # [B, 512, 1]

        # Transformer: [B, S, C]
        x = l4_points.permute(0, 2, 1)      # [B, 1, 512]
        x = self.transformer_encoder(x)     # [B, 1, 512]

        # Global pooling (S = 1)
        global_x = x.view(B, 512)

        x = self.drop1(F.relu(self.bn1(self.fc1(global_x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=-1)

        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        return F.nll_loss(pred, target)