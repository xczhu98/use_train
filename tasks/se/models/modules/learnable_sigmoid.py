"""SE 幅度解码器共用的可学习 sigmoid 激活。"""

# 来源：https://github.com/yxlu-0102/MP-SENet/blob/main/utils.py

import torch
import torch.nn as nn


class LearnableSigmoid2D(nn.Module):
    """逐频点学习斜率，保留 checkpoint 中的 slope 参数及其 [F,1] 形状。"""

    def __init__(self, in_features, beta=1):
        super(LearnableSigmoid2D, self).__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        """对最后两个轴为频率、时间的输入应用带斜率的 sigmoid。"""
        return self.beta * torch.sigmoid(self.slope * x)
