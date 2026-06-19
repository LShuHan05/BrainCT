import torch
import torch.nn as nn

# ====================== 时空注意力（原版） ======================
class SpatialTemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

# ====================== 【新增】SE 注意力模块 ======================
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ====================== 残差双卷积（可选 SE） ======================
class ResidualDoubleConv(nn.Module):
    def __init__(self, in_c, out_c, use_se=False, se_reduction=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
        )
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1),
                nn.BatchNorm2d(out_c)
            )
        self.se = SELayer(out_c, reduction=se_reduction) if use_se else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = out + residual
        out = self.relu(out)
        out = self.se(out)
        return out

# ====================== 深度监督辅助分类器 ======================
class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# ====================== 【新】多尺度特征融合模块 ======================
class MultiScaleFusion(nn.Module):
    """将不同尺度的编码器特征上采样后融合"""
    def __init__(self, in_ch_list, out_ch):
        super().__init__()
        self.upsample_convs = nn.ModuleList()
        for in_ch in in_ch_list:
            self.upsample_convs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
        self.final_conv = nn.Conv2d(out_ch * len(in_ch_list), out_ch, kernel_size=1)

    def forward(self, features):
        # features: list of tensors [d1, d2, d3, d4] from encoder
        target_size = features[0].shape[2:]
        fused = []
        for i, feat in enumerate(features):
            if feat.shape[2:] != target_size:
                feat = nn.functional.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            feat = self.upsample_convs[i](feat)
            fused.append(feat)
        fused = torch.cat(fused, dim=1)
        out = self.final_conv(fused)
        return out

# ====================== 增强版 UNet2D 模型 ======================
class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, use_auxiliary=True, use_se=True, se_reduction=16):
        super().__init__()
        print("\n🎉 增强版 UNet2D：SE 注意力 + 多尺度特征融合 + 深度监督")

        self.use_auxiliary = use_auxiliary
        self.pool = nn.MaxPool2d(2)

        # 编码器（残差卷积 + 可选 SE）
        self.d1 = ResidualDoubleConv(in_ch, 64, use_se=use_se, se_reduction=se_reduction)
        self.d2 = ResidualDoubleConv(64, 128, use_se=use_se, se_reduction=se_reduction)
        self.d3 = ResidualDoubleConv(128, 256, use_se=use_se, se_reduction=se_reduction)
        self.d4 = ResidualDoubleConv(256, 512, use_se=use_se, se_reduction=se_reduction)

        # 时空注意力（瓶颈层）
        self.spatial_temporal_attn = SpatialTemporalAttention(dim=512, num_heads=4)

        # 【新增】多尺度特征融合（用于辅助监督）
        self.ms_fusion = MultiScaleFusion(in_ch_list=[64, 128, 256, 512], out_ch=256)

        # 深度监督辅助分类器
        if self.use_auxiliary:
            self.aux_d3 = AuxiliaryClassifier(256, out_ch)
            self.aux_d4 = AuxiliaryClassifier(512, out_ch)
            self.aux_fusion = AuxiliaryClassifier(256, out_ch)   # 从融合特征输出

        # 解码器
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.u4 = ResidualDoubleConv(512, 256, use_se=use_se, se_reduction=se_reduction)
        self.u3 = ResidualDoubleConv(256, 128, use_se=use_se, se_reduction=se_reduction)
        self.u2 = ResidualDoubleConv(128, 64, use_se=use_se, se_reduction=se_reduction)

        self.out = nn.Conv2d(64, out_ch, kernel_size=1)

        self.fused_feature = None

    def forward(self, x):
        # 编码器
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))

        # 瓶颈层注意力
        d4 = self.spatial_temporal_attn(d4)

        # 多尺度融合特征
        fused = self.ms_fusion([d1, d2, d3, d4])

        # 深度监督输出
        aux_outputs = {}
        if self.training and self.use_auxiliary:
            aux_outputs['aux_d4'] = self.aux_d4(d4)
            aux_outputs['aux_d3'] = self.aux_d3(d3)
            aux_outputs['aux_fusion'] = self.aux_fusion(fused)   # 新监督

        # 解码器
        u4 = self.u4(torch.cat([self.up4(d4), d3], dim=1))
        u3 = self.u3(torch.cat([self.up3(u4), d2], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), d1], dim=1))

        out = self.out(u2)

        # 推理时特征提取
        if not self.training:
            f1 = torch.mean(d1, dim=(2, 3))
            f2 = torch.mean(d2, dim=(2, 3))
            f3 = torch.mean(d3, dim=(2, 3))
            f4 = torch.mean(d4, dim=(2, 3))
            feat = torch.cat([f1, f2, f3, f4], dim=1)
            self.fused_feature = torch.nn.functional.normalize(feat, p=2, dim=1)

        if self.training and self.use_auxiliary:
            return {'main': out, **aux_outputs}
        return out

    def extract_features(self):
        if self.fused_feature is None:
            raise ValueError("请先执行一次 model(x) 推理，再调用 extract_features()")
        return self.fused_feature