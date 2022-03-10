from typing import Any
from model import UnetBase
import torch
import torch.nn as nn
from model.attention.CBAM import CBAM


class ResBlockWithAttention(nn.Module):
    """
    带注意力机制的残差块
    """
    expansion = 1

    def __init__(self, in_channels, out_channels):
        super(ResBlockWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        self.attention = CBAM(out_channels, 16)
        if in_channels is not out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

    def _forward_unimplemented(self, *x: Any) -> None:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            identity = self.downsample(identity)
        out = self.attention(out)
        out += identity
        out = self.relu(out)
        return out

    def forward(self, x):
        """
        前向
        @param x:
        @return:
        """
        return self._forward_unimplemented(x)


class AttentionUnet(UnetBase):
    """
    注意力机制的Unet,利用cbam通道注意力和空间注意力结合的注意力机制
    """

    def __init__(self, in_channels, out_channels):
        super(UnetBase, self).__init__(in_channels, out_channels)

    def forward(self, x):
        """
        前向输出
        @param x:
        """
        return self._forward_unimplemented(x)

    @staticmethod
    def base_block(in_channels, out_channels):
        """
        基本构成块，残差结构
        @param in_channels:
        @param out_channels:
        """
        block = ResBlockWithAttention(in_channels, out_channels)
        return block

    @staticmethod
    def contract_block(in_channels, out_channels):
        """
        压缩
        @param in_channels:输入通道数
        @param out_channels:输出通道数
        @return:
        """
        block = torch.nn.Sequential(
            AttentionUnet.base_block(in_channels, out_channels),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        return block

    @staticmethod
    def expansive_block(in_channels, mid_channels, out_channels):
        """
        扩大
        @param in_channels:输入通道数
        @param mid_channels:中间通道数
        @param out_channels:输出通道数
        @return:
        """
        block = torch.nn.Sequential(
            AttentionUnet.base_block(in_channels, mid_channels),
            torch.nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1),
        )
        return block

    @staticmethod
    def bottom_block(in_channels, mid_channels, out_channels):
        """
        瓶颈层
        @param in_channels:输入通道数
        @param mid_channels:中间通道数
        @param out_channels:输出通道数
        @return:
        """
        block = torch.nn.Sequential(
            AttentionUnet.base_block(in_channels, mid_channels),
            torch.nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)
        )
        return block

    @staticmethod
    def final_block(in_channels, mid_channels, out_channels):
        """
        最后的输出层
        @param in_channels:输入通道数
        @param mid_channels:中间通道数
        @param out_channels:输出通道数
        @return:
        """
        block = torch.nn.Sequential(
            AttentionUnet.base_block(in_channels, mid_channels),
            torch.nn.Conv2d(kernel_size=1, in_channels=mid_channels, out_channels=out_channels)
        )
        return block
