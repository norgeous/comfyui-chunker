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


def decode_video(video_vae, video_t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Decode a [B, 24, T, H, W] H3 video latent to [N, H, W, C] frames in [0, 1].

    Returns None when the VAE is missing, the latent is not a video stream, or
    the decode fails for any reason (the caller falls back to taeh3 / rgb).
    """
    if video_vae is None or video_t is None or video_t.dim() != 5:
        return None
    try:
        with torch.no_grad():
            decoded = video_vae.decode(video_t)  # [B, T, H, W, C] float32 [0, 1]
        if decoded.dim() == 4:
            decoded = decoded.unsqueeze(1)
        if decoded.dim() != 5:
            return None
        return decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1]).float().clamp(0, 1)
    except Exception:
        return None


def decode_audio(audio_vae, audio_t: Optional[torch.Tensor]) -> Optional[dict]:
    """Decode a [B, 32, 2, T] H3 audio latent to an AUDIO dict at the VAE sample rate.

    Decodes stereo [B, 2, L] waveforms in [-1, 1]. Returns None when the VAE or
    stream is missing, or the decode fails for any reason.
    """
    if audio_vae is None or audio_t is None or audio_t.dim() != 4:
        return None
    try:
        sample_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        with torch.no_grad():
            decoded = audio_vae.decode(audio_t)  # [B, L, 2]
        waveform = decoded.movedim(-1, 1).contiguous()  # [B, 2, L]
        if waveform.dim() != 3:
            return None
        return {"waveform": waveform, "sample_rate": sample_rate}
    except Exception:
        return None


def decode_av_latent(latent, video_vae=None, audio_vae=None):
    """Decode an H3 AV latent with the real VAE pair.

    Returns (images [N, H, W, C] in [0, 1], audio dict | None), or (None, None)
    when the VAEs or matching stream(s) are unavailable so callers can fall back
    to taeh3 / rgb.
    """
    if video_vae is None or latent is None:
        return None, None
    samples = latent.get("samples") if isinstance(latent, dict) else latent
    if samples is None:
        return None, None

    video_t = None
    audio_t = None
    if hasattr(samples, "tensors"):
        for t in samples.tensors:
            if t.dim() == 5 and video_t is None:
                video_t = t
            elif t.dim() == 4 and audio_t is None:
                audio_t = t
    else:
        video_t = samples

    images = decode_video(video_vae, video_t)
    if images is None:
        return None, None
    audio_dict = decode_audio(audio_vae, audio_t) if audio_t is not None else None
    return images, audio_dict