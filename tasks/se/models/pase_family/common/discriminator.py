"""GAN Discriminators for PASE: MPD + MBD.

References:
  - MultiPeriodDiscriminator: BigVGAN (NVIDIA)
  - MultiBandDiscriminator: Vocos / descript-audio-codec
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from torchaudio.transforms import Spectrogram


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class DiscriminatorP(nn.Module):
    """Single-period discriminator."""

    def __init__(
        self,
        period: int,
        channel_mult: int = 1,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        self.d_mult = channel_mult
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, int(32 * self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(32 * self.d_mult), int(128 * self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(128 * self.d_mult), int(512 * self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(512 * self.d_mult), int(1024 * self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(1024 * self.d_mult), int(1024 * self.d_mult), (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(int(1024 * self.d_mult), 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator (MPD)."""

    def __init__(
        self,
        mpd_reshapes: List[int] = None,
        channel_mult: int = 1,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        if mpd_reshapes is None:
            mpd_reshapes = [2, 3, 5, 7, 11]
        self.mpd_reshapes = mpd_reshapes
        self.discriminators = nn.ModuleList([
            DiscriminatorP(rs, channel_mult, use_spectral_norm=use_spectral_norm)
            for rs in self.mpd_reshapes
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]
    ]:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorB(nn.Module):
    """Single-band STFT discriminator."""

    def __init__(
        self,
        window_length: int,
        channels: int = 32,
        hop_factor: float = 0.25,
        bands: Tuple[Tuple[float, float], ...] = (
            (0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)
        ),
    ):
        super().__init__()
        self.window_length = window_length
        self.hop_factor = hop_factor
        self.spec_fn = Spectrogram(
            n_fft=window_length,
            hop_length=int(window_length * hop_factor),
            win_length=window_length,
            power=None,
        )
        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands
        convs = lambda: nn.ModuleList([
            weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))),
        ])
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

    def spectrogram(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = x - x.mean(dim=-1, keepdims=True)
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        x = self.spec_fn(x)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 2, 1)  # [B, F, T, C] -> [B, C, T, F]
        x_bands = [x[..., b[0]:b[1]] for b in self.bands]
        return x_bands

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x_bands = self.spectrogram(x.squeeze(1))
        fmap = []
        x_out = []
        for band, stack in zip(x_bands, self.band_convs):
            for i, layer in enumerate(stack):
                band = layer(band)
                band = F.leaky_relu(band, 0.1)
                if i > 0:
                    fmap.append(band)
            x_out.append(band)
        x_out = torch.cat(x_out, dim=-1)
        x_out = self.conv_post(x_out)
        fmap.append(x_out)
        return x_out, fmap


class MultiBandDiscriminator(nn.Module):
    """Multi-Band Multi-Scale STFT Discriminator (MBD)."""

    def __init__(self, mbd_fft_sizes: List[int] = None):
        super().__init__()
        if mbd_fft_sizes is None:
            mbd_fft_sizes = [2048, 1024, 512]
        self.fft_sizes = mbd_fft_sizes
        self.discriminators = nn.ModuleList([DiscriminatorB(window_length=w) for w in self.fft_sizes])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]
    ]:
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class CombinedDiscriminator(nn.Module):
    """Combined MPD + MBD discriminator wrapper."""

    def __init__(self, mpd_reshapes=None, mbd_fft_sizes=None, channel_mult=1, use_spectral_norm=False):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(
            mpd_reshapes=mpd_reshapes, channel_mult=channel_mult, use_spectral_norm=use_spectral_norm
        )
        self.mbd = MultiBandDiscriminator(mbd_fft_sizes=mbd_fft_sizes)

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]
    ]:
        y_d_rs_mpd, y_d_gs_mpd, fmap_rs_mpd, fmap_gs_mpd = self.mpd(y, y_hat)
        y_d_rs_mbd, y_d_gs_mbd, fmap_rs_mbd, fmap_gs_mbd = self.mbd(y, y_hat)
        y_d_rs = y_d_rs_mpd + y_d_rs_mbd
        y_d_gs = y_d_gs_mpd + y_d_gs_mbd
        fmap_rs = fmap_rs_mpd + fmap_rs_mbd
        fmap_gs = fmap_gs_mpd + fmap_gs_mbd
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
