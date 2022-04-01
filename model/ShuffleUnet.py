from model import UnetBase
import torch


class ShuffleUnet(UnetBase):
    """
    结合通道洗牌操作的卷积模块结合Unet，在保证准确率的前提下，减少参数量
    """

    def __init__(self, in_channel, out_channels):
        super(ShuffleUnet, self).__init__(in_channel, out_channels)

    def forward(self, x):
        """
        前向输出
        @param x:
        """
        return self._forward_unimplemented(x)

    @staticmethod
    def base_block(in_channels, out_channels):
        """
        基本构成块，
        @param in_channels:
        @param out_channels:
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU6(),
            torch.nn.Conv2d(in_channels, out_channels, 1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU6(),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU6(),
            torch.nn.Conv2d(out_channels, out_channels, 1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU6(),
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
            ShuffleUnet.base_block(in_channels, out_channels),
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
            ShuffleUnet.base_block(in_channels, mid_channels),
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
            ShuffleUnet.base_block(in_channels, mid_channels),
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
            ShuffleUnet.base_block(in_channels, mid_channels),
            torch.nn.Conv2d(kernel_size=1, in_channels=mid_channels, out_channels=out_channels)
        )
        return block
