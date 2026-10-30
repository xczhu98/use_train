"""GAP-URGENet (Cisco, 2025) generative-and-predictive speech enhancement package.

Ported from https://github.com/Xiaobin-Rong/gap-urgenet（官方仓库）。五模块流水线：

    生成支路 (16k): DeWavLM-Omni (WavLM + PLC) -> VocosAdapter -> VocosVocoder
    预测支路 (16k): Predictor (单通道 TFGridNetV3, predictor/tfgridnet.py)
    融合     (48k): PostNet (双通道 TFGridNetV3 + BandSplit, postnet/tfgrid_cws.py)

复用本框架共享组件：unipase 包的 WavLM_feat_PLC / VocosAdapter / VocosVocoder，
以及 unipase.postnet.tfgrid_cws_res 的 GridNetV3Block / BandSplit /
LayerNormalization / AllHeadPReLULayerNormalization4DC（与 GAP-URGENet 官方
models/predictor/tfgridnet.py、models/postnet/tfgrid_cws.py 中同名类逐行比对
确认数值行为完全一致，仅去 espnet 依赖）。
"""
