"""GAP-URGENet PostNet 包：48kHz 双通道融合网络（TFGridNetV3 + BandSplit）。"""

from .tfgrid_cws import TFGridNet

__all__ = ["TFGridNet"]
