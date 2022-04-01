
# -*- coding: utf-8 -*-

import torch
from typing import Any
from abc import abstractmethod
import torch.nn as nn


class LossBase(nn.Module):
    """
    分割常用损失函数
    """
    def __init__(self):
        super(LossBase, self).__init__()


