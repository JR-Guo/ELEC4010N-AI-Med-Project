import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out):
        super(ConvBlock, self).__init__()

        ops = []
        input_channel = n_filters_in
        for i in range(n_stages):
            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            ops.append(nn.BatchNorm3d(n_filters_out))
            ops.append(nn.ReLU(inplace=True))
            input_channel = n_filters_out

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride),
            nn.BatchNorm3d(n_filters_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride),
            nn.BatchNorm3d(n_filters_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Model(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16):
        super().__init__()
        self.block_1 = ConvBlock(1, n_channels, n_filters)
        self.block_1_dw = DownsamplingConvBlock(n_filters, 2 * n_filters)

        self.block_2 = ConvBlock(2, n_filters * 2, n_filters * 2)
        self.block_2_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4)

        self.block_3 = ConvBlock(3, n_filters * 4, n_filters * 4)
        self.block_3_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8)

        self.block_4 = ConvBlock(3, n_filters * 8, n_filters * 8)
        self.block_4_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16)

        self.block_5 = ConvBlock(3, n_filters * 16, n_filters * 16)
        self.block_5_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8)

        self.block_6 = ConvBlock(3, n_filters * 8, n_filters * 8)
        self.block_6_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4)

        self.block_7 = ConvBlock(3, n_filters * 4, n_filters * 4)
        self.block_7_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2)

        self.block_8 = ConvBlock(2, n_filters * 2, n_filters * 2)
        self.block_8_up = UpsamplingDeconvBlock(n_filters * 2, n_filters)

        self.block_9 = ConvBlock(1, n_filters, n_filters)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def encoder(self, input):
        x1 = self.block_1(input)
        x1_dw = self.block_1_dw(x1)

        x2 = self.block_2(x1_dw)
        x2_dw = self.block_2_dw(x2)

        x3 = self.block_3(x2_dw)
        x3_dw = self.block_3_dw(x3)

        x4 = self.block_4(x3_dw)
        x4_dw = self.block_4_dw(x4)

        x5 = self.block_5(x4_dw)
        return (x1, x2, x3, x4, x5)

    def decoder(self, features):
        x1, x2, x3, x4, x5 = features

        x5_up = self.block_5_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_6(x5_up)
        x6_up = self.block_6_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_7(x6_up)
        x7_up = self.block_7_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_8(x7_up)
        x8_up = self.block_8_up(x8)
        x8_up = x8_up + x1

        x9 = self.block_9(x8_up)
        out = self.out_conv(x9)
        return out

    def forward(self, input):
        features = self.encoder(input)
        out = self.decoder(features)
        return out
