"""DeepFilterNet 统一入口：基线结构 + YAML 参数化变体。

本文件为 DeepFilterNet 的**对外接口层**，底层实现见 `deepfilternet_impl.py`。

文件职责：
  - 模型注册与配置路由
  - 采样率转换（支持动态 fs）
  - 参数验证与文档

因果性说明：
  DPDFNet 固定保留 `conv_lookahead=2` 和 `df_lookahead=2`，两者对应 2 帧未来上下文，
  因而属于 **near-causal**（非严格因果）。cache 可支持实时逐帧处理，但只能复用历史状态，
  不能消除这 2 帧前瞻；在 16 kHz、hop_length=160 时，固定算法前瞻约为 20 ms。

底层实现见：
  - deepfilternet_impl.py: STFT/ERB/DPRNN/DeepFiltering 等 32 个模块

GRU 变体说明（``gru_variant`` 选择 GRU 类与 skip 位置）：
  - ``squeezed_s``（默认）：SqueezedGRU_S，skip 在 ``linear_out`` 之后；DFN3/DPDFNet 使用该变体。
  - ``squeezed``：SqueezedGRU，skip 在 GRU 输出上；复现原版 DeepFilterNet2 时使用。

模型变体配置：
  - DeepFilterNet2（原版）: enc_concat=true, enc_lin_groups=8, lin_groups=8, erb_apply_above_bins=null, dprnn_num_blocks=0，gru_variant=**squeezed**
  - DeepFilterNet3: enc_concat=false, enc_lin_groups=32, lin_groups=16, erb_apply_above_bins=96, dprnn_num_blocks=0；**采样率为 48kHz**（STFT 常用 n_fft=960, hop=480）
  - DPDFNet baseline / 变体: enc_concat=true, enc_lin_groups=32, lin_groups=16, erb_apply_above_bins=null, dprnn_num_blocks=0|2|4|8，gru_variant=squeezed_s

配置示例：
  model:
    type: DeepFilterNet
    args:
      samplerate: 16000
      dprnn_num_blocks: 4  # 0=DF2/DF3, 2/4/8=DPDFNet变体
      enc_concat: true
      enc_lin_groups: 32
      lin_groups: 16
      erb_apply_above_bins: null  # null=全频ERB, 96=仅高频(DF3)
"""

# When run directly, add project root to sys.path
if __name__ == "__main__":
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))

import torch

from tasks.se.models.base_model import BaseSEModel
from core.utils.registry import MODEL_REGISTRY


def _build_deepfilternet_impl(samplerate: int, dprnn_num_blocks: int, **kw):
    """构建底层 DeepFilterNet 实现。
    
    底层实现包含 32 个模块（STFT/ERB/DPRNN/DF 等），见 deepfilternet_impl.py。
    """
    from tasks.se.models.modules.deepfilternet_impl import DeepFilterNet as _Impl
    return _Impl(samplerate=samplerate, dprnn_num_blocks=dprnn_num_blocks, **kw)


@MODEL_REGISTRY.register("DeepFilterNet")
class DeepFilterNet(BaseSEModel):
    """DeepFilterNet 统一模型：支持 DF2/DF3/DPDFNet 变体。

    通过 YAML model.args 配置不同变体：
      - DeepFilterNet2（原版）: dprnn_num_blocks=0, enc_concat=true, enc_lin_groups=8, lin_groups=8, erb_apply_above_bins=null，gru_variant='squeezed'
      - DeepFilterNet3: dprnn_num_blocks=0, enc_concat=false, enc_lin_groups=32, lin_groups=16, erb_apply_above_bins=96；官方 48kHz（STFT 960/480）
      - DPDFNet baseline: dprnn_num_blocks=0, enc_lin_groups=32, lin_groups=16，gru_variant='squeezed_s'（2.15M）；DPDFNet-2/4/8: dprnn_num_blocks=2/4/8

    参数说明：
      samplerate: 采样率。DF2/DPDFNet 常用 16000；DeepFilterNet3 官方为 48000（对应 STFT n_fft=960, hop_length=480）
      dprnn_num_blocks: DPRNN 块数（0=DF2/DF3, 2/4/8=DPDFNet 变体）
      enc_concat: Encoder 拼接方式（true=DF2/DPDFNet, false=DF3）
      enc_lin_groups/lin_groups: GroupedLinear 的 group 数
      erb_apply_above_bins: ERB 作用频段（null=全频, 96=仅高频~4.8kHz以上）
      gru_variant: GRU 变体。'squeezed_s'=DFN3/DPDFNet（skip 在 linear_out 后），'squeezed'=原版 DFN2（skip 在 GRU 输出上）。复现原版 DeepFilterNet2 可设 gru_variant='squeezed'；DeepFilterNet3/DPDFNet 用 'squeezed_s'（默认）。
    """

    def __init__(
        self,
        samplerate: int = 16000,
        dprnn_num_blocks: int = 0,
        enc_concat: bool = True,
        enc_lin_groups: int = 32,
        lin_groups: int = 16,
        erb_apply_above_bins: int | None = None,
        gru_variant: str = 'squeezed_s',
        **kw
    ):
        super().__init__()
        self.model_sr = samplerate
        # 与 BSRNN 等一致，供 audit / profile 在未写 data.sample_rate 时选用原生采样率（DF3=48000）
        self.target_fs = int(samplerate)
        n_fft = kw.pop("n_fft", None)
        hop_length = kw.pop("hop_length", None)
        win_length = kw.pop("win_length", None)
        # window/center 属于 runtime-owned STFT；底层模型不消费，但必须从
        # build_model 注入的参数中显式消费，避免被误判为未知配置。
        kw.pop("window", None)
        kw.pop("center", None)
        if kw:
            raise TypeError(
                f"DeepFilterNet got unexpected keyword arguments: {sorted(kw)}"
            )
        impl_kwargs = {}
        if n_fft is not None:
            impl_kwargs["n_fft"] = n_fft
        if hop_length is not None:
            impl_kwargs["hop_length"] = hop_length
        if win_length is not None:
            impl_kwargs["win_length"] = win_length
        self._impl = _build_deepfilternet_impl(
            samplerate=samplerate,
            dprnn_num_blocks=dprnn_num_blocks,
            enc_concat=enc_concat,
            enc_lin_groups=enc_lin_groups,
            lin_groups=lin_groups,
            erb_apply_above_bins=erb_apply_above_bins,
            gru_variant=gru_variant,
            **impl_kwargs,
        )
        self.n_fft = getattr(self._impl, "n_fft", 320)
    def forward(
        self,
        noisy_spec: torch.Tensor,
        lengths=None,
        fs: int = None,
        return_lsnr: bool = False,
        return_alpha: bool = False,
    ):
        """Spectrum-in / spectrum-out. STFT/iSTFT must be done by the caller.
        DeepFilterNet uses n_fft=320 (frame length 320); use that in the caller STFT.

        Args:
            noisy_spec: (B, F, T, 2) real-valued complex spectrum (F = n_fft//2+1, n_fft=320).
            lengths: (B,) optional.
            fs: ignored (kept for API compatibility).
            return_lsnr: if True, return (enhanced_spec, lsnr).
            return_alpha: if True, return (enhanced_spec, lsnr, df_alpha); requires return_lsnr.

        Returns:
            enhanced_spec: (B, F, T, 2), or (enhanced_spec, lsnr) if return_lsnr,
            or (enhanced_spec, lsnr, df_alpha) if return_alpha.
        """
        if noisy_spec.dim() != 4 or noisy_spec.shape[-1] != 2:
            raise ValueError(
                "DeepFilterNet accepts only spectrum input (B, F, T, 2). "
                "STFT 须与训练一致：16k 常用 n_fft=320；DeepFilterNet3（48k）常用 n_fft=960。"
            )
        if return_alpha and not return_lsnr:
            raise ValueError("return_alpha=True requires return_lsnr=True")
        if return_alpha:
            enhanced_spec, lsnr, df_alpha = self._impl.forward_from_spec(
                noisy_spec, return_alpha=True
            )
            # 零系数锚定：辅助输出以 0 权重接入计算图，保持梯度/dtype 链路；
            # nan_to_num 需同时归零 ±inf，否则 0.0 * inf = nan 会污染增强谱。
            anchor = lsnr.nan_to_num(0.0, posinf=0.0, neginf=0.0).sum()
            anchor = anchor + df_alpha.nan_to_num(0.0, posinf=0.0, neginf=0.0).sum()
            enhanced_spec = enhanced_spec + 0.0 * anchor
            return enhanced_spec, lsnr, df_alpha
        enhanced_spec, lsnr = self._impl.forward_from_spec(noisy_spec)
        enhanced_spec = enhanced_spec + 0.0 * lsnr.nan_to_num(0.0, posinf=0.0, neginf=0.0).sum()
        if return_lsnr:
            return enhanced_spec, lsnr
        return enhanced_spec


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 验证 DeepFilterNet2、DeepFilterNet3、DPDFNet2/4/8 参数量与算力，与 tools/model/audit_metrics.py 及 config 表一致。
    # DeepFilterNet3 官方为 48kHz，此处用 48k + STFT 960/480；其余为 16k + STFT 320/160。
    # 运行: python tasks/se/models/deepfilternet.py [--sr 16000] [--dur 1.0] [--device cpu]
    # -------------------------------------------------------------------------
    import torch.nn as nn
    from core.utils.model_profile import profile_model, make_profile_parser
    from core.stft import build_stft_istft

    args = make_profile_parser(default_sr=16000).parse_args()

    STFT_16K = {"n_fft": 320, "hop_length": 160, "win_length": 320, "window": "hann"}
    STFT_48K = {"n_fft": 960, "hop_length": 480, "win_length": 960, "window": "hann"}
    # 与 configs 表、audit_metrics.py 对照的官方 Params[M] / MACs[G]（论文/DPDFNet 表）
    OFFICIAL = {
        "DeepFilterNet2": (2.31, 0.36),  # 原版（Rikorose），论文 +Simplified DNN
        "DeepFilterNet3": (None, None),
        "DPDFNet2": (2.49, 1.35),
        "DPDFNet4": (2.84, 2.36),
        "DPDFNet8": (3.54, 4.37),
    }
    # (label, model_args, sample_rate, stft_cfg)
    VARIANTS = [
        ("DeepFilterNet2", {"samplerate": 16000, "dprnn_num_blocks": 0, "enc_concat": True, "enc_lin_groups": 8, "lin_groups": 8, "erb_apply_above_bins": None, "gru_variant": "squeezed"}, 16000, STFT_16K),
        ("DeepFilterNet3", {"samplerate": 48000, "dprnn_num_blocks": 0, "enc_concat": False, "enc_lin_groups": 32, "lin_groups": 16, "erb_apply_above_bins": 96, "n_fft": 960, "hop_length": 480, "win_length": 960}, 48000, STFT_48K),
        ("DPDFNet2", {"samplerate": 16000, "dprnn_num_blocks": 2, "enc_concat": True, "enc_lin_groups": 32, "lin_groups": 16, "erb_apply_above_bins": None}, 16000, STFT_16K),
        ("DPDFNet4", {"samplerate": 16000, "dprnn_num_blocks": 4, "enc_concat": True, "enc_lin_groups": 32, "lin_groups": 16, "erb_apply_above_bins": None}, 16000, STFT_16K),
        ("DPDFNet8", {"samplerate": 16000, "dprnn_num_blocks": 8, "enc_concat": True, "enc_lin_groups": 32, "lin_groups": 16, "erb_apply_above_bins": None}, 16000, STFT_16K),
    ]

    class _W(nn.Module):
        def __init__(self, m, stft_m):
            super().__init__()
            self._m, self._stft = m, stft_m
        def forward(self, wav):
            spec = self._stft(wav)
            out_spec, _ = self._m(spec, return_lsnr=True)
            return self._stft.inverse(out_spec, wav.shape[-1])

    print("\n[Official] Params[M] / MACs[G]:", OFFICIAL)
    for label, model_args, sr, stft_cfg in VARIANTS:
        stft_module = build_stft_istft(stft_cfg)
        model = DeepFilterNet(**model_args)
        wrapped = _W(model, stft_module)
        official_p, official_g = OFFICIAL.get(label, (None, None))
        print("\n[Model]", label, "| sr:", sr, "| STFT:", stft_cfg, "| official:", f"{official_p}M / {official_g}G" if official_p else "—")
        ret = profile_model(
            wrapped,
            sample_rate=sr,
            duration=args.dur,
            batch_size=args.batch,
            device=args.device,
            repeats=args.repeats,
            label=label,
        )
        if official_p and ret:
            p = ret["params"] / 1e6
            g = (ret["macs"] / 1e9) if ret.get("macs") else None
            if g is not None:
                print(f"  -> params {p:.4f}M (official {official_p}M)  macs {g:.4f}G (official {official_g}G)")
            else:
                print(f"  -> params {p:.4f}M (official {official_p}M)")
