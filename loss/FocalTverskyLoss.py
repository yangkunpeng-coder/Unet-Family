
# -*- coding: utf-8 -*-
from loss.LossBase import LossBase
from abc import abstractmethod
from torch import Tensor
import torch.nn.functional as F
import torch
import torch.nn as nn


class FocalTverskyLoss(LossBase):
    """
    Dice Loss with bec_weight
    """

    def __init__(self):
        super(FocalTverskyLoss, self).__init__()

    @staticmethod
    def __tversky(prediction: Tensor, target: Tensor) -> Tensor:
        """
        tversky 标准
        @param prediction:预测
        @param target:真实
        @return:
        """
        smooth = 1
        y_true_pos = torch.flatten(target)
        y_pred_pos = torch.flatten(prediction)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

    @staticmethod
    def __focal_tversky(prediction: Tensor, target: Tensor) -> Tensor:
        """
        cal focal tversky
        @param prediction:预测
        @param target:真实
        @return:
        """
        y_true = target.type(torch.float32)
        y_pred = prediction.type(torch.float32)
        pt_1 = FocalTverskyLoss.__tversky(y_true, y_pred)
        gamma = 0.75
        return torch.pow((1 - pt_1), gamma)

    @staticmethod
    def __tversky_loss(prediction: Tensor, target: Tensor) -> Tensor:
        return 1 - FocalTverskyLoss.__tversky(prediction, target)

    @staticmethod
    @abstractmethod
    def cal_loss(prediction: Tensor, target: Tensor) -> Tensor:
        """
        计算损失
        @param prediction:
        @param target:
        """
        bce = F.binary_cross_entropy_with_logits(prediction, target)
        prediction = F.sigmoid(prediction)
        loss = FocalTverskyLoss.__tversky_loss(prediction, target)
        return loss

