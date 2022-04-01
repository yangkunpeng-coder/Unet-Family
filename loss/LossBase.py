# -*- coding: utf-8 -*-

import torch
from torch import Tensor
from typing import Any
from abc import abstractmethod
import torch.nn as nn
from typing import Optional


class LossBase(nn.Module):
    """
    分割常用损失函数
    """

    def __init__(self):
        super(LossBase, self).__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """

        @param input:
        @param target:
        @return:
        """
        return self.cal_loss(input, target)

    @staticmethod
    @abstractmethod
    def cal_loss(input: Tensor, target: Tensor) -> Tensor:
        """
        计算损失
        @param input:
        @param target:
        """
        pass
