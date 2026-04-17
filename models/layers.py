import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as Taudio
import torchaudio.functional as Faudio


class LogMelSpectrogramLayer(nn.Module):
    """Implement a custom layer that processes a normalized Log-Mel-Spectrogram analysis."""
    def __init__(self, sample_rate=44100, n_fft=2048, win_length=None, hop_length=512, n_mels=128, f_min=20, f_max=None, center=True):
        super(LogMelSpectrogramLayer, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.center = center

        self.mel_scale = Taudio.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            center=self.center,
        )

        self.amplitude_to_db = Taudio.AmplitudeToDB(stype='power')

    def min_max_normalize(self, tensor: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
        """Normalizes the input tensor to the specified range [min_val, max_val]."""
        min_tensor = torch.as_tensor(min_val, dtype=tensor.dtype, device=tensor.device)
        max_tensor = torch.as_tensor(max_val, dtype=tensor.dtype, device=tensor.device)
        eps = 1e-9

        t_min = tensor.min()
        t_max = tensor.max()

        t_range = t_max - t_min + eps
        normalized_tensor = (tensor - t_min) / t_range

        scaled_tensor = normalized_tensor * (max_tensor - min_tensor) + min_tensor
        return scaled_tensor

    def forward(self, x):
        x = self.mel_scale(x)
        x = self.amplitude_to_db(x)
        x = self.min_max_normalize(x)
        return x.to(torch.float32)


class LogMelSpectrogramLayerONNX(nn.Module):
    """ONNX-compatible Log-Mel-Spectrogram layer using manual STFT pipeline."""
    def __init__(self, sample_rate=44100, n_fft=2048, win_length=None, hop_length=512, n_mels=128, f_min=20, f_max=None, center=True):
        super(LogMelSpectrogramLayerONNX, self).__init__()
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length
        self.center = center

        self.register_buffer('window', torch.hann_window(self.win_length))

        f_max_val = float(f_max) if f_max is not None else float(sample_rate // 2)
        mel_fb = Faudio.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=float(f_min),
            f_max=f_max_val,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        self.register_buffer('mel_fb', mel_fb)  # (n_freqs, n_mels)

    def min_max_normalize(self, tensor: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
        """Normalizes the input tensor to the specified range [min_val, max_val]."""
        min_tensor = torch.as_tensor(min_val, dtype=tensor.dtype, device=tensor.device)
        max_tensor = torch.as_tensor(max_val, dtype=tensor.dtype, device=tensor.device)
        eps = 1e-9

        t_min = tensor.min()
        t_max = tensor.max()

        t_range = t_max - t_min + eps
        normalized_tensor = (tensor - t_min) / t_range

        scaled_tensor = normalized_tensor * (max_tensor - min_tensor) + min_tensor
        return scaled_tensor

    def forward(self, x):
        # x: (batch, 1, samples)
        x = x.squeeze(1)  # (batch, samples)

        if self.center:
            x = F.pad(x, (self.n_fft // 2, self.n_fft // 2), mode='reflect')

        # STFT with return_complex=False → (batch, n_fft//2+1, time_frames, 2)
        stft = torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            self.window, center=False, return_complex=False
        )

        # Power spectrum: real² + imag² → (batch, n_fft//2+1, time_frames)
        power = stft[..., 0] ** 2 + stft[..., 1] ** 2

        # Apply mel filterbank: (n_freqs, n_mels).T @ (batch, n_freqs, time_frames)
        # → (batch, n_mels, time_frames)
        mel = torch.matmul(self.mel_fb.T, power)

        # Power to dB, top_db=80
        mel_db = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        mel_db = torch.max(mel_db, mel_db.max() - 80.0)

        # Add channel dim: (batch, 1, n_mels, time_frames)
        mel_db = mel_db.unsqueeze(1)

        mel_db = self.min_max_normalize(mel_db)
        return mel_db.to(torch.float32)


class AdaptivePool2dONNX(nn.Module):
    """ONNX-compatible replacement for nn.AdaptiveAvgPool2d."""
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = tuple(output_size)

    def forward(self, x: torch.Tensor):
        h_in, w_in = x.shape[-2], x.shape[-1]
        h_out, w_out = self.output_size

        pad_h = max(0, h_out - h_in)
        pad_w = max(0, w_out - w_in)

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            h_in, w_in = x.shape[-2], x.shape[-1]

        stride_h = max(1, h_in // h_out)
        stride_w = max(1, w_in // w_out)

        kernel_h = h_in - (h_out - 1) * stride_h
        kernel_w = w_in - (w_out - 1) * stride_w

        return F.avg_pool2d(
            x,
            kernel_size=(int(kernel_h), int(kernel_w)),
            stride=(int(stride_h), int(stride_w))
        )


class customConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, padding):
        super(customConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.batch = nn.BatchNorm2d(output_channels)
        self.activ = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.activ(x)
        return x
