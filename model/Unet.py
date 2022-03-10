import torch
from model import UnetBase


class UNet(UnetBase):
    """
    输入和输出分辨率相同的Unet网络
    """

    def __init__(self, in_channel, out_channels):
        super(UNet, self).__init__(in_channel, out_channels)

    def forward(self, x):
        """
        前向
        @param x:
        @return:
        """
        return self._forward_unimplemented(x)

    @staticmethod
    def base_block(in_channels, out_channels):
        """
        基本构成块
        @param in_channels:
        @param out_channels:
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
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
            UNet.base_block(in_channels, out_channels),
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
            UNet.base_block(in_channels, mid_channels),
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
            UNet.base_block(in_channels, mid_channels),
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
            UNet.base_block(in_channels, mid_channels),
            torch.nn.Conv2d(kernel_size=1, in_channels=mid_channels, out_channels=out_channels)
        )
        return block
