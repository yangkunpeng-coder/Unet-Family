# -*- coding: utf-8 -*-
from loss.LossBase import LossBase
from abc import abstractmethod
from torch import Tensor
import torch.nn.functional as F
import torch
import torch.nn as nn


class DiceLoss(LossBase):
    """
    Dice Loss with bec_weight
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    @staticmethod
    def __dice_loss(prediction: Tensor, target: Tensor) -> Tensor:
        smooth = 1.0
        i_flat = prediction.view(-1)
        t_flat = target.view(-1)
        intersection = (i_flat * t_flat).sum()
        return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))

    @staticmethod
    @abstractmethod
    def cal_loss(prediction: Tensor, target: Tensor, bce_weight=0.5) -> Tensor:
        """
        计算损失
        @param prediction:
        @param target:
        @param bce_weight:
        """
        bce = F.binary_cross_entropy_with_logits(prediction, target)
        prediction = F.sigmoid(prediction)
        dice = DiceLoss.__dice_loss(prediction, target)
        loss = bce * bce_weight + dice * (1 - bce_weight)
        return loss
