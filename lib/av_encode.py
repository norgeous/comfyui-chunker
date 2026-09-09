from typing import Optional

import torch
import torchaudio


def enforce_stereo(waveform: torch.Tensor) -> torch.Tensor:
    """Return a [B, 2, L] waveform, duplicating a mono [B, 1, L] channel to stereo."""
    if waveform.shape[1] == 2:
        return waveform
    return waveform.repeat(1, 2, 1)


def encode_video(video_vae, images: torch.Tensor) -> torch.Tensor:
    """Encode [N, H, W, C] frames in [0, 1] to a video latent.

    MiniMax H3 video VAE: [N, H, W, C] -> [1, 24, T_lat, H/16, W/16].
    """
    return video_vae.encode(images)


def encode_audio(audio_vae, audio: dict) -> torch.Tensor:
    """Encode an AUDIO dict to a latent, resampling to the VAE's sample rate.

    MiniMax H3 audio VAE: stereo [B, 2, L] at 32 kHz -> [1, 32, 2, T_lat].
    """
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sample_rate != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sr)
    waveform = enforce_stereo(waveform)
    return audio_vae.encode(waveform.movedim(1, -1))


def pack_av_latent(video_t: torch.Tensor, audio_t: Optional[torch.Tensor] = None):
    """Pack video (+ optional audio) latent streams into an H3-style NestedTensor.

    Returns the raw tensor / NestedTensor (the caller wraps it in a LATENT dict).
    """
    from comfy.nested_tensor import NestedTensor

    if audio_t is None:
        return video_t
    return NestedTensor((video_t, audio_t))