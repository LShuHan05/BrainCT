import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        print("\n🎉 普通UNet2D模型！")
        self.pool = nn.MaxPool2d(2)
        self.d1 = DoubleConv(in_ch, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.u4 = DoubleConv(512, 256)
        self.u3 = DoubleConv(256, 128)
        self.u2 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))

        u4 = self.u4(torch.cat([self.up4(d4), d3], dim=1))
        u3 = self.u3(torch.cat([self.up3(u4), d2], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), d1], dim=1))
        out = self.out(u2)
        return out