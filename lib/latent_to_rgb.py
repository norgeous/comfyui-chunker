import math
from typing import Optional

import torch
import torch.nn.functional as F

# MiniMax H3 video VAE temporal constants (comfy/ldm/minimax/vae.py)
_CLIP_LENGTH = 17
_VAE_RATIO_T = 4
_TOKEN_DROP = 3
_TOKENS_CHUNK_SIZE = math.ceil(_CLIP_LENGTH / _VAE_RATIO_T)  # 5
_TOKEN_OVERLAP = (-_TOKEN_DROP) % _TOKENS_CHUNK_SIZE          # 2
_FRAME_PRE_PADDING = (-_CLIP_LENGTH) % _VAE_RATIO_T           # 3
_CHUNK_DEC = _TOKENS_CHUNK_SIZE * _VAE_RATIO_T                # 20
_SPLIT_COUNT = int(_TOKEN_DROP > 0) + 1                       # 2


def _h3_temporal_chunks(token_count: int):
    pseudo_total = token_count + _TOKEN_DROP
    pad_tokens = (-pseudo_total) % _TOKENS_CHUNK_SIZE
    pseudo_total += pad_tokens
    num_chunks = pseudo_total // _TOKENS_CHUNK_SIZE - int(_TOKEN_DROP > 0)
    if num_chunks < 1:
        pad_tokens += _TOKENS_CHUNK_SIZE
        num_chunks += 1
    return pad_tokens, num_chunks


def _h3_temporal_pad_frames(z_len: int, pad_tokens: int) -> int:
    if pad_tokens <= 0:
        return 0
    intra_tail = _CLIP_LENGTH % _VAE_RATIO_T
    z_len_before_pad = z_len - pad_tokens
    return sum(
        intra_tail if (z_len_before_pad + k) % _TOKENS_CHUNK_SIZE == 0
        else _VAE_RATIO_T
        for k in range(pad_tokens)
    )


def _h3_decode_frames(token_count: int) -> int:
    if token_count == 1:
        return 1
    pad_tokens, num_chunks = _h3_temporal_chunks(token_count)
    z_len = token_count + pad_tokens
    total_frames = 0
    final_overlap_frames = 0
    for i in range(num_chunks):
        t_start_idx = i * _TOKENS_CHUNK_SIZE
        t_end_idx = t_start_idx + _TOKENS_CHUNK_SIZE + _TOKEN_OVERLAP
        clip_token_len = max(0, min(t_end_idx, z_len) - min(t_start_idx, z_len))
        clip_frame_len = clip_token_len * _VAE_RATIO_T
        for j in range(_SPLIT_COUNT):
            f_start_idx = j * _CHUNK_DEC
            f_end_idx = min(f_start_idx + _CHUNK_DEC, clip_frame_len)
            chunk_frames = max(0, f_end_idx - f_start_idx - _FRAME_PRE_PADDING)
            if j == 0:
                total_frames += chunk_frames
            else:
                final_overlap_frames = chunk_frames
    return total_frames + final_overlap_frames - _h3_temporal_pad_frames(z_len, pad_tokens)


def _frames_from_tokens(token_count: int, mode: str) -> int:
    if mode == "minimax-h3":
        return _h3_decode_frames(token_count)
    if mode == "wan2":
        return max(1, (token_count - 1) * 4 + 1)
    if mode == "ltx2":
        return max(1, (token_count - 1) * 8 + 1)
    return token_count


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
        b, t = frames.shape[0], 1

    rgb = F.linear(frames, factors, bias)
    rgb = ((rgb + 1.0) / 2.0).clamp(0, 1)

    target_frames = b * _frames_from_tokens(t, mode)
    if rgb.shape[0] != target_frames:
        if target_frames == 1:
            idx = torch.zeros(1, dtype=torch.long, device=rgb.device)
        else:
            src = rgb.shape[0]
            idx = (torch.arange(target_frames, device=rgb.device) * (src - 1) / (target_frames - 1)).round().long()
        rgb = rgb[idx]

    if target_w is None:
        target_w = video.shape[-1] * latent_format.spacial_downscale_ratio
    if target_h is None:
        target_h = video.shape[-2] * latent_format.spacial_downscale_ratio

    if rgb.shape[2] != target_w or rgb.shape[1] != target_h:
        from .utils_tensor import resize_image
        rgb = resize_image(rgb, target_w, target_h)

    return rgb