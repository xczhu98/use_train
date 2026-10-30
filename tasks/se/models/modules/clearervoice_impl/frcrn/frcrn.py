import torch.nn as nn
import torch 
import torch.nn.functional as F
import numpy as np
from .conv_stft import ConvSTFT, ConviSTFT
from .unet import UNet

class FRCRN_SE_16K(nn.Module):
    def __init__(self, args):
        super(FRCRN_SE_16K, self).__init__()
        self.model = DCCRN(
            complex=True,
            model_complexity=45,
            model_depth=14,
            log_amp=False,
            padding_mode="zeros",
            win_len=args.win_len,
            win_inc=args.win_inc,
            fft_len=args.fft_len,
            win_type=args.win_type
        )

    def forward(self, x):
        output = self.model(x)
        return output[1][0]

class DCCRN(nn.Module):
    def __init__(self, complex, model_complexity, model_depth, log_amp, padding_mode, win_len=400, win_inc=100, fft_len=512, win_type='hanning'):
        """
        :param complex: whether to use complex networks.
        :param model_complexity: only used for model_depth 20
        :param model_depth: Only two options are available : 14 or 20
        :param log_amp: Whether to use log amplitude to estimate signals
        :param padding_mode: Encoder's convolution filter. 'zeros', 'reflect'
        """
        super().__init__()
        self.feat_dim = fft_len // 2 +1

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        fix = True
        self.stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='complex', fix=fix)
        self.unet = UNet(1, complex=complex, model_complexity=model_complexity, model_depth=model_depth, padding_mode=padding_mode)
        self.unet2 = UNet(1, complex=complex, model_complexity=model_complexity, model_depth=model_depth, padding_mode=padding_mode)

    def forward(self, inputs):
        # 上游 waveform 进/出路径：会绕过顶层 stft 参数契约与 checkpoint 对齐假设，
        # 框架内一律走 forward_enhance_spec 的谱进谱出路径，这里显式拒绝误用。
        raise RuntimeError(
            "DCCRN.forward (waveform in/out) is disabled: use forward_enhance_spec "
            "with framework STFT spectra (B, D, T, 2)"
        )

    def forward_enhance_spec(self, noisy_spec: torch.Tensor) -> torch.Tensor:
        """Framework STFT layout: ``noisy_spec`` (B, D, T, 2) real/last -> enhanced (B, D, T, 2).

        Skips internal ConvSTFT/ConviSTFT; UNet + mask apply only. ``D`` must equal ``feat_dim``.
        """
        if noisy_spec.dim() != 4 or noisy_spec.size(-1) != 2:
            raise ValueError(f"Expected noisy_spec (B, D, T, 2), got {tuple(noisy_spec.shape)}")
        _b, d, _t, _ = noisy_spec.shape
        if d != self.feat_dim:
            raise ValueError(
                f"freq bins D={d} != feat_dim={self.feat_dim} (set YAML stft.n_fft so n_fft//2+1 == {self.feat_dim})"
            )
        real = noisy_spec[..., 0]
        imag = noisy_spec[..., 1]
        cmp_stacked = torch.cat([real, imag], dim=1).unsqueeze(1)
        cmp_spec = torch.cat(
            [cmp_stacked[:, :, : self.feat_dim, :], cmp_stacked[:, :, self.feat_dim :, :]],
            dim=1,
        )
        cmp_spec = torch.unsqueeze(cmp_spec, 4)
        cmp_spec = torch.transpose(cmp_spec, 1, 4)
        unet1_out = self.unet(cmp_spec)
        cmp_mask1 = torch.tanh(unet1_out)
        unet2_out = self.unet2(unet1_out)
        cmp_mask2 = torch.tanh(unet2_out)
        cmp_mask2 = cmp_mask2 + cmp_mask1
        est_line = torch.cat(
            [
                cmp_spec[:, :, :, :, 0] * cmp_mask2[:, :, :, :, 0]
                - cmp_spec[:, :, :, :, 1] * cmp_mask2[:, :, :, :, 1],
                cmp_spec[:, :, :, :, 0] * cmp_mask2[:, :, :, :, 1]
                + cmp_spec[:, :, :, :, 1] * cmp_mask2[:, :, :, :, 0],
            ],
            dim=1,
        )
        est_line = torch.cat([est_line[:, 0, :, :], est_line[:, 1, :, :]], dim=1)
        er = est_line[:, : self.feat_dim, :]
        ei = est_line[:, self.feat_dim :, :]
        return torch.stack([er, ei], dim=-1)

    def inference(self, inputs):
        # 与 forward 同一类 waveform 便捷路径：绕过顶层 stft 契约，显式拒绝。
        raise RuntimeError(
            "DCCRN.inference (waveform in/out) is disabled: use forward_enhance_spec "
            "with framework STFT spectra (B, D, T, 2)"
        )

    def apply_mask(self, cmp_spec, cmp_mask):
        est_spec = torch.cat([cmp_spec[:,:,:,:,0]*cmp_mask[:,:,:,:,0]-cmp_spec[:,:,:,:,1]*cmp_mask[:,:,:,:,1], cmp_spec[:,:,:,:,0]*cmp_mask[:,:,:,:,1]+cmp_spec[:,:,:,:,1]*cmp_mask[:,:,:,:,0]],1)
        est_spec = torch.cat([est_spec[:,0,:,:], est_spec[:,1,:,:]], 1)
        cmp_mask = torch.squeeze(cmp_mask, 1)
        cmp_mask = torch.cat([cmp_mask[:,:,:,0], cmp_mask[:,:,:,1]],1)
        est_wav = self.istft(est_spec)
        est_wav = torch.squeeze(est_wav, 1)
        return est_spec, est_wav, cmp_mask

    def get_params(self, weight_decay=0.0):
            # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        return params 
