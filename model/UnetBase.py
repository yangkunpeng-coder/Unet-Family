from typing import Any
from abc import abstractmethod
import torch.nn as nn
import torch


class UnetBase(nn.Module):
    """
    Unet的基类,因为unet算法的结构相同，所以使用23种设计模式中的《模板方法》
    """

    def __init__(self, in_channel, out_channels):
        super(UnetBase, self).__init__()
        # Encode
        self.encode1 = self.contract_block(in_channels=in_channel, out_channels=16)
        self.encode2 = self.contract_block(in_channels=16, out_channels=32)
        self.encode3 = self.contract_block(in_channels=32, out_channels=64)
        self.encode4 = self.contract_block(in_channels=64, out_channels=128)

        # Bottleneck
        self.bottleneck = self.bottom_block(in_channels=128, mid_channels=256, out_channels=128)

        # Decode
        self.decode1 = self.expansive_block(in_channels=256, mid_channels=128, out_channels=64)
        self.decode2 = self.expansive_block(in_channels=128, mid_channels=64, out_channels=32)
        self.decode3 = self.expansive_block(in_channels=64, mid_channels=32, out_channels=16)

        # output
        self.final = self.final_block(in_channels=32, mid_channels=16, out_channels=out_channels)

    def _forward_unimplemented(self, *input: Any) -> None:
        """
        Encode
        @param x:
        @return:
        """
        encode1 = self.encode1(input)
        encode2 = self.encode2(encode1)
        encode3 = self.encode3(encode2)
        encode4 = self.encode4(encode3)

        # Bottleneck
        bottleneck = self.bottleneck(encode4)

        # Decode
        concat1 = self.concat(encode4, bottleneck)
        decode1 = self.decode1(concat1)
        concat2 = self.concat(encode3, decode1)
        decode2 = self.decode2(concat2)
        concat3 = self.concat(encode2, decode2)
        decode3 = self.decode3(concat3)
        concat4 = self.concat(encode1, decode3)
        net = self.final(concat4)
        return net

    @staticmethod
    @abstractmethod
    def base_block(in_channels, out_channels):
        """
        Unet中基本的模块
        @param in_channels: 输入通道
        @param out_channels:输出通道
        """
        pass

    @staticmethod
    @abstractmethod
    def contract_block(in_channels, out_channels):
        """
        压缩
        @param in_channels:
        @param out_channels:
        """
        pass

    @staticmethod
    @abstractmethod
    def expansive_block(in_channels, mid_channels, out_channels):
        """
        扩大
        @param in_channels:输入通道数
        @param mid_channels:中间通道数
        @param out_channels:输出通道数
        @return:
        """
        pass

    @staticmethod
    @abstractmethod
    def final_block(in_channels, mid_channels, out_channels):
        """
        最后的输出层
        @param in_channels:输入通道数
        @param mid_channels:中间通道数
        @param out_channels:输出通道数
        @return:
        """
        pass

    @staticmethod
    def concat(down_sample, up_sample):
        """
        拼接函数
        @param down_sample:
        @param up_sample:
        @return:
        """
        return torch.cat((down_sample, up_sample), 1)
