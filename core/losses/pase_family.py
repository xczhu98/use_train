"""Loss functions for PASE-family (pase_family) models.

Includes:
  - PaseMultiScaleMelLoss: multi-scale mel spectrogram L1 distance.
  - MSELoss: simple wrapper for Phase-2 distillation.
  - ScalarPassthroughLoss: compatibility re-export from the lightweight module.
  - GAPHybridLoss: GAP-URGENet Predictor 训练损失（压缩谱 RI + Mag + 带符号 SDR 变体）。
  - feature_loss / discriminator_loss / generator_loss: GAN losses for the
    waveform-domain MPD+MBD discriminator (``PASEDiscriminator``), used by
    ``tasks.se.trainer.pase_gan_trainer.PASEGANTrainer``. Ported verbatim from
    ``pase/utils/loss.py`` in the source repo.
"""

import functools
import math
import sys
from collections import namedtuple
from pathlib import Path
from typing import List, Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from scipy import signal
from torchaudio.functional import resample

from core.losses.utils import create_frame_mask, prepare_centered_stft_inputs
from core.losses.scalar_passthrough import ScalarPassthroughLoss
from core.utils.registry import LOSS_REGISTRY
from core.utils.safe_load import safe_torch_load


@LOSS_REGISTRY.register("PaseMultiScaleMelLoss")
class PaseMultiScaleMelLoss(nn.Module):
    """Multi-scale mel spectrogram loss adapted from BigVGAN/descript-audio-codec.

    Default 7 scales: window_lengths=[32,64,128,256,512,1024,2048],
    n_mels=[5,10,20,40,80,160,320]. ``lengths`` limits each scale to valid
    centered-STFT frames, so batch padding contributes neither loss nor gradient.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        n_mels: List[int] = None,
        window_lengths: List[int] = None,
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        pow: float = 1.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = None,
        mel_fmax: List[float] = None,
        window_type: str = "hann",
    ):
        super().__init__()
        if n_mels is None:
            n_mels = [5, 10, 20, 40, 80, 160, 320]
        if window_lengths is None:
            window_lengths = [32, 64, 128, 256, 512, 1024, 2048]
        if mel_fmin is None:
            mel_fmin = [0.0] * len(window_lengths)
        if mel_fmax is None:
            mel_fmax = [None] * len(window_lengths)
        schedules = {
            "n_mels": tuple(n_mels),
            "window_lengths": tuple(window_lengths),
            "mel_fmin": tuple(mel_fmin),
            "mel_fmax": tuple(mel_fmax),
        }
        lengths = {len(values) for values in schedules.values()}
        if lengths == {0}:
            raise ValueError("PaseMultiScaleMelLoss schedules must not be empty")
        if len(lengths) != 1:
            raise ValueError(
                "PaseMultiScaleMelLoss schedules must have equal lengths; "
                + ", ".join(f"{name}={len(values)}" for name, values in schedules.items())
            )
        for index, (n_mel, window_length, fmin, fmax) in enumerate(
            zip(
                schedules["n_mels"],
                schedules["window_lengths"],
                schedules["mel_fmin"],
                schedules["mel_fmax"],
            )
        ):
            if type(n_mel) is not int or n_mel <= 0:
                raise ValueError(f"n_mels[{index}] must be an exact positive integer")
            if type(window_length) is not int or window_length <= 0:
                raise ValueError(f"window_lengths[{index}] must be an exact positive integer")
            if type(fmin) not in (int, float) or not math.isfinite(fmin) or fmin < 0:
                raise ValueError(f"mel_fmin[{index}] must be finite and non-negative")
            if fmax is not None and (
                type(fmax) not in (int, float)
                or not math.isfinite(fmax)
                or fmax <= fmin
                or fmax > sampling_rate / 2
            ):
                raise ValueError(
                    f"mel_fmax[{index}] must be None or finite in (mel_fmin, Nyquist]"
                )

        self.sampling_rate = sampling_rate
        STFTParams = namedtuple("STFTParams", ["window_length", "hop_length", "window_type", "match_stride"])
        self.stft_params = [
            STFTParams(window_length=w, hop_length=w // 4, match_stride=match_stride, window_type=window_type)
            for w in schedules["window_lengths"]
        ]
        self.n_mels = schedules["n_mels"]
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = schedules["mel_fmin"]
        self.mel_fmax = schedules["mel_fmax"]
        self.pow = pow
        self.loss_fn = nn.L1Loss()

    @staticmethod
    @functools.lru_cache(None)
    def get_window(window_type, window_length):
        return signal.get_window(window_type, window_length)

    @staticmethod
    @functools.lru_cache(None)
    def get_mel_filters(sr, n_fft, n_mels, fmin, fmax):
        return librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def mel_spectrogram(
        self,
        wav,
        n_mels,
        fmin,
        fmax,
        window_length,
        hop_length,
        match_stride,
        window_type,
        lengths=None,
    ):
        B, C, T = wav.shape
        channel_lengths = None
        if lengths is not None:
            channel_lengths = lengths.to(wav.device).repeat_interleave(C)
        wav = wav.reshape(B * C, T)
        if channel_lengths is not None:
            wav = wav[:, : int(channel_lengths.max().item())]
        (wav,), channel_lengths, _ = prepare_centered_stft_inputs(
            wav, lengths=channel_lengths, n_fft=window_length
        )
        if match_stride:
            assert hop_length == window_length // 4
            right_pad = math.ceil(T / hop_length) * hop_length - T
            pad = (window_length - hop_length) // 2
            wav = F.pad(wav, (pad, pad + right_pad), mode="reflect")

        window = self.get_window(window_type, window_length)
        window = torch.from_numpy(window).to(wav.device).float()
        stft = torch.stft(
            wav,
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        if match_stride:
            stft = stft[..., 2:-2]
        _, nf, nt = stft.shape
        magnitude = torch.abs(stft).reshape(B, C, nf, nt)
        mel_basis = self.get_mel_filters(
            self.sampling_rate, 2 * (nf - 1), n_mels, fmin, fmax
        )
        mel_basis = torch.from_numpy(mel_basis).to(wav.device)
        mel_spectrogram = magnitude.transpose(2, -1) @ mel_basis.T
        mel_spectrogram = mel_spectrogram.transpose(-1, 2)
        frame_mask = create_frame_mask(
            channel_lengths,
            nt,
            window_length,
            window_length,
            hop_length,
            wav.device,
        ).reshape(B, C, 1, nt)
        if match_stride:
            frame_mask = frame_mask[..., :nt]
        return mel_spectrogram, frame_mask

    @staticmethod
    def _masked_l1(x: torch.Tensor, y: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        mask = frame_mask.to(x.dtype)
        denominator = (mask.sum() * x.shape[-2]).clamp_min(1.0)
        return ((x - y).abs() * mask).sum() / denominator

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths=None) -> torch.Tensor:
        if enhanced.ndim == 2:
            enhanced = enhanced.unsqueeze(1)
        if clean.ndim == 2:
            clean = clean.unsqueeze(1)
        loss = 0.0
        for n_mels, fmin, fmax, s in zip(self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params):
            kwargs = {
                "n_mels": n_mels, "fmin": fmin, "fmax": fmax,
                "window_length": s.window_length, "hop_length": s.hop_length,
                "match_stride": s.match_stride, "window_type": s.window_type,
                "lengths": lengths,
            }
            x_mels, frame_mask = self.mel_spectrogram(enhanced, **kwargs)
            y_mels, _ = self.mel_spectrogram(clean, **kwargs)
            log10 = torch.log(x_mels.new_tensor(10.0))
            x_logmels = torch.log(x_mels.clamp(min=self.clamp_eps).pow(self.pow)) / log10
            y_logmels = torch.log(y_mels.clamp(min=self.clamp_eps).pow(self.pow)) / log10
            loss += self.log_weight * self._masked_l1(x_logmels, y_logmels, frame_mask)
            loss += self.mag_weight * self._masked_l1(x_mels, y_mels, frame_mask)
        return loss * self.weight


class _UTMOSAwareLoss(nn.Module):
    """冻结 UTMOS22 Strong 参数，仅保留 waveform→MOS 的输入梯度。"""

    def __init__(self, model_dir: str | None = None):
        super().__init__()
        root = Path(model_dir).expanduser() if model_dir else Path(__file__).resolve().parents[2] / "third_party_scripts/UTMOS"
        source = root / "SpeechMOS"
        weights = root / "utmos22_strong_step7459_v1.pt"
        if not (source / "speechmos").is_dir() or not weights.is_file():
            raise FileNotFoundError(f"UTMOS source/weights not found under {root}")
        # 只在该次导入窗口内提升 vendored 源码路径，导入后立即移除，
        # 避免永久改变进程级模块解析顺序（speechmos 已进入 sys.modules）。
        inserted = str(source) not in sys.path
        if inserted:
            sys.path.insert(0, str(source))
        try:
            from speechmos.utmos22.strong.model import UTMOS22Strong
        except ImportError as exc:
            raise RuntimeError("UTMOS-aware loss 无法导入 vendored SpeechMOS") from exc
        finally:
            if inserted:
                try:
                    sys.path.remove(str(source))
                except ValueError:
                    pass
        self.model = UTMOS22Strong()
        state = safe_torch_load(weights, map_location="cpu", purpose="load UTMOS-aware loss weights")
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, enhanced: torch.Tensor, clean=None, lengths=None) -> torch.Tensor:
        return -self.model(enhanced.float(), 16000).mean()


@LOSS_REGISTRY.register("GAPPostNetAwareLoss")
class GAPPostNetAwareLoss(nn.Module):
    """GAP-URGENet PostNet 的 Mel + PESQ/UTMOS-aware 重建损失。

    GAP-URGENet 论文要求 PostNet 在 Vocoder 损失上增加 PESQ/UTMOS-aware 项；
    其引用的 TS-URGENet 使用可微 PESQ 与冻结 UTMOS 模型，两项权重均为 0.5。
    本实现默认两项权重为 0，保持既有 YAML 的纯 Mel 数值；启用时逐样本按
    ``lengths`` 裁剪并可微重采样到两个代理要求的 16 kHz。
    """

    def __init__(
        self,
        sampling_rate: int = 48000,
        pesq_weight: float = 0.0,
        utmos_weight: float = 0.0,
        allow_missing_aware: bool = False,
        utmos_model_dir: str | None = None,
        **mel_kwargs,
    ):
        super().__init__()
        if pesq_weight < 0 or utmos_weight < 0:
            raise ValueError("PESQ/UTMOS-aware loss weights must be non-negative")
        self.sampling_rate = int(sampling_rate)
        self.pesq_weight = float(pesq_weight)
        self.utmos_weight = float(utmos_weight)
        self.allow_missing_aware = bool(allow_missing_aware)
        self.utmos_model_dir = utmos_model_dir
        self.mel_loss = PaseMultiScaleMelLoss(sampling_rate=sampling_rate, **mel_kwargs)
        self.pesq_loss = self._build_pesq_loss() if self.pesq_weight else None
        self.utmos_loss = self._build_utmos_loss() if self.utmos_weight else None

    def _build_pesq_loss(self) -> nn.Module:
        from core.losses.perceptual_metric import PESQLoss
        return PESQLoss(sample_rate=16000, mode="wb", allow_missing=self.allow_missing_aware)

    def _build_utmos_loss(self) -> nn.Module | None:
        try:
            return _UTMOSAwareLoss(self.utmos_model_dir)
        except (ImportError, FileNotFoundError, RuntimeError):
            if self.allow_missing_aware:
                return None
            raise

    @staticmethod
    def _mono(wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 3 and wav.shape[1] == 1:
            return wav[:, 0]
        if wav.ndim != 2:
            raise ValueError(f"GAPPostNetAwareLoss expects (B,T) or (B,1,T), got {tuple(wav.shape)}")
        return wav

    def forward(self, enhanced: torch.Tensor, clean: torch.Tensor, lengths=None) -> torch.Tensor:
        total = self.mel_loss(enhanced, clean, lengths)
        if self.pesq_loss is None and self.utmos_loss is None:
            return total
        enhanced, clean = self._mono(enhanced), self._mono(clean)
        metric_total = enhanced.sum() * 0.0
        for index in range(enhanced.shape[0]):
            length = int(lengths[index].item()) if lengths is not None else enhanced.shape[-1]
            enh = enhanced[index : index + 1, :length]
            ref = clean[index : index + 1, :length]
            if self.sampling_rate != 16000:
                enh = resample(enh, self.sampling_rate, 16000)
                ref = resample(ref, self.sampling_rate, 16000)
            if self.pesq_loss is not None:
                metric_total = metric_total + self.pesq_weight * self.pesq_loss(enh, ref)
            if self.utmos_loss is not None:
                metric_total = metric_total + self.utmos_weight * self.utmos_loss(enh, ref)
        return total + metric_total / enhanced.shape[0]


@LOSS_REGISTRY.register("MSELoss")
class MSELoss(nn.MSELoss):
    """Simple MSE wrapper for Phase-2 DeWavLM distillation.

    Accepts ``lengths`` and extra kwargs for compatibility with the framework's
    validation loop (``MetricsCapableTrainer._validation_epoch``), which passes
    ``lengths`` to every loss function.
    """

    def forward(self, input, target, lengths=None, **kwargs):
        return super().forward(input, target)


@LOSS_REGISTRY.register("GAPHybridLoss")
class GAPHybridLoss(nn.Module):
    """GAP-URGENet Predictor 训练损失。

    声明 ``REQUIRES_SAMPLE_RATE``：框架多采样率训练时由 Trainer 在每个 batch
    注入实际 ``sample_rate``，STFT 的 n_fft/hop/win 随 batch 采样率换算；单采样率
    配置仍使用构造参数 ``sample_rate`` 作为缺省值。

    Ported from ``gap-urgenet/utils/loss_factory.py:33-89``（官方 HybridLoss），
    数学公式逐行对齐，未做任何"修正"：

      - 压缩复数谱 RI 损失：S_c = S / |S|^(1-c)，c=compress_factor=0.3，
        对 real/imag 分别取 MSE，权重 lamda_ri=30；
      - 压缩幅度损失：(|S_p|^c - |S_t|^c)^2 的 MSE，权重 lamda_mag=70；
      - 带符号 SDR 变体：-2*log10(‖pred‖ / ‖pred-true‖)，权重 lamda_sdr=1。
        注意这不是标准 SI-SDR（无零均值化、无最优缩放、系数为 2），
        官方注释虽写 "SISNR loss" 但公式如此，照搬。

    STFT 参数与 Predictor 模型内部一致：按时长比例参数化
    （fft_len=0.032s, hop_len=0.016s, win_len=0.032s, hann 窗），
    16kHz 下即 n_fft=512 / hop=256。

    与官方接口的差异：官方 forward(y_pred, y_true, fs) 逐样本传 fs；
    本框架通过 ``REQUIRES_SAMPLE_RATE`` 由 Trainer 在 forward 注入 batch
    实际 ``sample_rate``，未注入时回退构造参数 ``sample_rate``（YAML 缺省）。
    谱项仅按真实边界内的有效 STFT 帧归约，signed SDR 仅使用有效 samples，
    padding 不参与损失。
    """

    REQUIRES_SAMPLE_RATE = True

    def __init__(
        self,
        sample_rate: int = 16000,
        fft_len: float = 0.032,
        hop_len: float = 0.016,
        win_len: float = 0.032,
        compress_factor: float = 0.3,
        eps: float = 1e-12,
        lamda_ri: float = 30,
        lamda_mag: float = 70,
        lamda_sdr: float = 1,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.fft_len = fft_len
        self.hop_len = hop_len
        self.win_len = win_len
        self.c = compress_factor
        self.eps = eps
        self.lamda_ri = lamda_ri
        self.lamda_mag = lamda_mag
        self.lamda_sdr = lamda_sdr

    def forward(self, y_pred, y_true, lengths=None, sample_rate=None, **kwargs):
        assert y_pred.shape[-1] == y_true.shape[-1]
        if y_pred.ndim == 3:
            y_pred = y_pred.squeeze(1)
        if y_true.ndim == 3:
            y_true = y_true.squeeze(1)

        device = y_true.device
        if sample_rate is None:
            fs = self.sample_rate
        else:
            if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
                raise TypeError(
                    "GAPHybridLoss.sample_rate must be an exact positive integer, "
                    f"got {sample_rate!r}"
                )
            if sample_rate <= 0:
                raise ValueError(
                    f"GAPHybridLoss.sample_rate must be positive, got {sample_rate!r}"
                )
            fs = int(sample_rate)

        n_fft = int(self.fft_len * fs)
        n_hop = int(self.hop_len * fs)
        n_win = int(self.win_len * fs)
        (pred_stft_input, true_stft_input), lengths, sample_mask = (
            prepare_centered_stft_inputs(
                y_pred, y_true, lengths=lengths, n_fft=n_fft
            )
        )
        window = torch.hann_window(n_win).to(device)
        pred_stft = torch.stft(
            pred_stft_input, n_fft, n_hop, n_win, window, return_complex=True
        )
        true_stft = torch.stft(
            true_stft_input, n_fft, n_hop, n_win, window, return_complex=True
        )

        pred_mag = torch.abs(pred_stft).clamp(self.eps)
        true_mag = torch.abs(true_stft).clamp(self.eps)

        pred_stft_c = pred_stft / pred_mag ** (1 - self.c)
        true_stft_c = true_stft / true_mag ** (1 - self.c)
        frame_mask = create_frame_mask(
            lengths,
            pred_stft.shape[-1],
            n_fft,
            n_win,
            n_hop,
            device,
        ).unsqueeze(1).to(pred_stft.real.dtype)
        valid_bins = (frame_mask.sum() * pred_stft.shape[1]).clamp(min=1.0)

        real_loss = (
            (pred_stft_c.real - true_stft_c.real).square() * frame_mask
        ).sum() / valid_bins
        imag_loss = (
            (pred_stft_c.imag - true_stft_c.imag).square() * frame_mask
        ).sum() / valid_bins
        mag_loss = (
            (pred_mag**self.c - true_mag**self.c).square() * frame_mask
        ).sum() / valid_bins

        # 官方 :80-85 的带符号 SDR 变体（非标准 SI-SDR，照搬公式）
        sample_mask = sample_mask.to(y_pred.dtype)
        sdr = -2 * torch.log10(
            torch.linalg.vector_norm(y_pred * sample_mask, dim=-1, keepdim=True)
            / torch.linalg.vector_norm(
                (y_pred - y_true) * sample_mask, dim=-1, keepdim=True
            ).clamp(self.eps)
            + self.eps
        ).mean()

        loss = self.lamda_ri * (real_loss + imag_loss) + self.lamda_mag * mag_loss + self.lamda_sdr * sdr
        return loss


# ---------------------------------------------------------------------------
# GAN losses for PASEDiscriminator (MPD + MBD, waveform-domain).
#
# These are plain functions, not LOSS_REGISTRY entries: they operate on
# List[Tensor] / List[List[Tensor]] (per-sub-discriminator score / feature-map
# lists), not the standard (enhanced, clean, lengths) -> scalar signature used
# by registered SE losses. Called directly from PASEGANTrainer.
# ---------------------------------------------------------------------------

def feature_loss(
    fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]
) -> torch.Tensor:
    """L1 distance between discriminator feature maps (real vs. generated)."""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[float], List[float]]:
    """LSGAN discriminator loss: mean((1-real)^2) + mean(fake^2), summed over sub-discs."""
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(
    disc_outputs: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """LSGAN generator loss: mean((1-fake)^2), summed over sub-discriminators."""
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses
