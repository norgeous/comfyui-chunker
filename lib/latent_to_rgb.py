from typing import Optional

import torch
import torch.nn.functional as F


def _pick_format(mode: str, channels: int):
    from comfy.latent_formats import MiniMaxH3Video, Wan21, Wan22, LTXV, SD15
    if mode == "minimax-h3":
        return MiniMaxH3Video()
    if mode == "wan2":
        return Wan22() if channels == 48 else Wan21()
    if mode == "ltx2":
        return LTXV()
    return SD15()


def _extract_video(latent) -> Optional[torch.Tensor]:
    if isinstance(latent, dict):
        latent = latent.get("samples")
    if latent is None:
        return None
    if hasattr(latent, "tensors"):
        for t in latent.tensors:
            if t.dim() == 5:
                return t
        return None
    if latent.dim() == 4 or latent.dim() == 5:
        return latent
    return None


def latent_to_images(latent, mode: str = "default", target_w: Optional[int] = None, target_h: Optional[int] = None) -> Optional[torch.Tensor]:
    video = _extract_video(latent)
    if video is None:
        return None

    latent_format = _pick_format(mode, video.shape[1])

    factors = torch.tensor(latent_format.latent_rgb_factors, device=video.device, dtype=video.dtype).transpose(0, 1)
    bias = None
    if latent_format.latent_rgb_factors_bias is not None:
        bias = torch.tensor(latent_format.latent_rgb_factors_bias, device=video.device, dtype=video.dtype)

    if video.dim() == 5:
        b, c, t, h, w = video.shape
        if t == 0:
            return None
        frames = video.permute(0, 2, 3, 4, 1).reshape(b * t, h, w, c)
    else:
        frames = video.movedim(1, -1).reshape(-1, *video.shape[2:], video.shape[1])

    rgb = F.linear(frames, factors, bias)
    rgb = ((rgb + 1.0) / 2.0).clamp(0, 1)

    if target_w is None:
        target_w = video.shape[-1] * latent_format.spacial_downscale_ratio
    if target_h is None:
        target_h = video.shape[-2] * latent_format.spacial_downscale_ratio

    if rgb.shape[2] != target_w or rgb.shape[1] != target_h:
        from .utils_tensor import resize_image
        rgb = resize_image(rgb, target_w, target_h)

    return rgb