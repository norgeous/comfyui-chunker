from typing import Any, Callable, Optional, Tuple

import torch


def pad_latent_to_length(latent: Any,
                         expected_video: int,
                         expected_audio: int,
                         empty_video_fn: Callable,
                         empty_audio_fn: Callable,
                         fallback_device: torch.device,
                         width: int = 0,
                         height: int = 0) -> Tuple[Optional[Any], int, int]:
    """Pad a chunk latent's video/audio streams up to their expected token counts.

    Streams shorter than expected are extended with `empty_*_fn` zero latents; streams
    the mode defines but that are missing entirely are created from zeros. Streams at or
    beyond their expected length are returned unchanged (never truncated).

    `latent` may be a plain video tensor or a NestedTensor (video + optional audio).
    An audio stream is kept / created only when `expected_audio > 0` (the mode carries
    an audio latent). Returns a plain video tensor when audio isn't expected, otherwise
    a NestedTensor with both streams.

    Returns (latent, video_tokens_added, audio_tokens_added).
    """
    video = None
    audio = None
    if latent is not None:
        if hasattr(latent, "tensors"):
            for t in latent.tensors:
                if video is None and t.dim() == 5:
                    video = t
                elif audio is None and t.dim() == 4:
                    audio = t
        elif latent.dim() == 5:
            video = latent

    video_pad = 0
    audio_pad = 0

    batch = 1
    if video is not None:
        batch = video.shape[0]
    elif audio is not None:
        batch = audio.shape[0]

    common_device = fallback_device
    if video is not None:
        common_device = video.device
    elif audio is not None:
        common_device = audio.device

    if expected_video > 0:
        if video is None:
            video = empty_video_fn(expected_video, width, height, batch, common_device)
            video_pad = expected_video
        elif video.shape[2] < expected_video:
            shortfall = expected_video - video.shape[2]
            pad = empty_video_fn(shortfall, width, height, video.shape[0], video.device)
            video = torch.cat([video, pad], dim=2)
            video_pad = shortfall

    if expected_audio > 0:
        if audio is None:
            audio = empty_audio_fn(expected_audio, batch, common_device)
            if audio is not None:
                audio_pad = expected_audio
        elif audio.shape[3] < expected_audio:
            shortfall = expected_audio - audio.shape[3]
            pad = empty_audio_fn(shortfall, audio.shape[0], audio.device)
            if pad is not None:
                audio = torch.cat([audio, pad], dim=3)
                audio_pad = shortfall
    else:
        audio = None

    if video is not None and audio is not None:
        from comfy.nested_tensor import NestedTensor
        return NestedTensor((video, audio)), video_pad, audio_pad
    return video if video is not None else audio, video_pad, audio_pad