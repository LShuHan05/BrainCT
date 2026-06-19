"""
脑部病灶多标签分类模型
支持：
1. 2D切片分类（单切片或三视图）
2. 3D体积分类
3. 多标签输出（9种病灶类型）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class LesionClassifier2D(nn.Module):
    """
    2D 多标签分类模型
    基于改进的 ResNet 架构
    """

    def __init__(
            self,
            num_classes: int = 9,
            input_channels: int = 1,
            use_three_views: bool = False
    ):
        super().__init__()

        self.use_three_views = use_three_views

        print(f"\n🎉 2D 病灶分类模型")
        print(f"   输入通道: {input_channels}")
        print(f"   类别数: {num_classes}")
        print(f"   三视图模式: {use_three_views}")

        # 特征提取 backbone
        self.backbone = self._create_backbone(input_channels)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头
        feature_dim = 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def _create_backbone(self, in_channels: int) -> nn.Module:
        """创建 ResNet-style backbone"""

        layers = []

        # 初始卷积层
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ))

        # Residual blocks
        layers.append(self._make_layer(64, 64, blocks=2, stride=1))
        layers.append(self._make_layer(64, 128, blocks=2, stride=2))
        layers.append(self._make_layer(128, 256, blocks=2, stride=2))
        layers.append(self._make_layer(256, 512, blocks=2, stride=2))

        return nn.Sequential(*layers)

    def _make_layer(
            self,
            in_channels: int,
            out_channels: int,
            blocks: int,
            stride: int
    ) -> nn.Sequential:
        """创建残差层"""
        layers = []

        # 第一个 block 可能需要下采样
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (B, C, H, W)

        Returns:
            logits: 分类 logits (B, num_classes)
        """
        # 特征提取
        features = self.backbone(x)

        # 全局池化
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)

        # 分类
        logits = self.classifier(pooled)

        return logits


class LesionClassifier3D(nn.Module):
    """
    3D 多标签分类模型
    处理完整体积数据
    """

    def __init__(self, num_classes: int = 9, input_channels: int = 1):
        super().__init__()

        print(f"\n🎉 3D 病灶分类模型")
        print(f"   输入通道: {input_channels}")
        print(f"   类别数: {num_classes}")

        # 3D CNN backbone
        self.backbone = self._create_3d_backbone(input_channels)

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 分类头
        feature_dim = 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()

    def _create_3d_backbone(self, in_channels: int) -> nn.Module:
        """创建 3D CNN backbone"""
        layers = []

        # 初始卷积
        layers.append(nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        ))

        # 3D Residual blocks
        layers.append(self._make_3d_layer(64, 64, blocks=2, stride=1))
        layers.append(self._make_3d_layer(64, 128, blocks=2, stride=2))
        layers.append(self._make_3d_layer(128, 256, blocks=2, stride=2))
        layers.append(self._make_3d_layer(256, 512, blocks=2, stride=2))

        return nn.Sequential(*layers)

    def _make_3d_layer(
            self,
            in_channels: int,
            out_channels: int,
            blocks: int,
            stride: int
    ) -> nn.Sequential:
        """创建 3D 残差层"""
        layers = []
        layers.append(ResidualBlock3D(in_channels, out_channels, stride))

        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入体积 (B, C, D, H, W)

        Returns:
            logits: 分类 logits (B, num_classes)
        """
        features = self.backbone(x)
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        logits = self.classifier(pooled)

        return logits


class ResidualBlock(nn.Module):
    """2D 残差块"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 快捷连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class ResidualBlock3D(nn.Module):
    """3D 残差块"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


# 向后兼容别名
LesionClassifier = LesionClassifier2D
