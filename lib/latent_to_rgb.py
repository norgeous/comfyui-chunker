import math
from typing import Optional

import torch
import torch.nn as nn
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


_taeh3_decoder = None


def _build_taeh3_decoder(sd: dict) -> nn.Sequential:
    """Rebuild the small 2D taeh3 decoder from its flat positional state dict.

    The `taeh3.safetensors` file in `models/vae_approx` is a 24-channel, 96-wide tiny
    autoencoder decoder with 4x spatial upsampling. Keys are positional module indices:
    `N.conv.0.weight` is a residual Block, `N.weight` a bare conv, and the gaps between
    indices are parameterless (Clamp at 0, ReLU at 2, upsamples elsewhere).
    """
    from comfy.taesd.taesd import Block, Clamp, conv

    by_index = {}
    for k, v in sd.items():
        head, _, rest = k.partition(".")
        if not head.isdigit():
            raise ValueError(f"not a flat TAE decoder state dict (unexpected key '{k}')")
        by_index.setdefault(int(head), {})[rest] = v

    modules = []
    for i in range(max(by_index) + 1):
        entry = by_index.get(i)
        if entry is None:
            modules.append(
                Clamp() if i == 0 else nn.ReLU() if i == 2 else nn.Upsample(scale_factor=2))
        elif "conv.0.weight" in entry:
            w = entry["conv.0.weight"]
            modules.append(Block(w.shape[1], w.shape[0]))
        elif "weight" in entry:
            w = entry["weight"]
            modules.append(conv(w.shape[1], w.shape[0], bias="bias" in entry))
        else:
            raise ValueError(f"unrecognized TAE decoder module at index {i}: {sorted(entry)}")
    return nn.Sequential(*modules)


def _get_taeh3_decoder() -> Optional[nn.Sequential]:
    global _taeh3_decoder
    if _taeh3_decoder is not None:
        return _taeh3_decoder
    try:
        import folder_paths
        path = folder_paths.get_full_path("vae_approx", "taeh3.safetensors")
        if path is None:
            return None
        import comfy.model_management
        import comfy.utils
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        decoder = _build_taeh3_decoder(sd)
        decoder.load_state_dict(sd)
        decoder = decoder.eval().to(comfy.model_management.vae_device(), comfy.model_management.vae_dtype())
        _taeh3_decoder = decoder
        return decoder
    except Exception:
        _taeh3_decoder = None
        return None


def latent_decode_taeh3(latent, mode: str = "default", max_tokens: Optional[int] = None) -> Optional[torch.Tensor]:
    """Decode a MiniMax H3 latent with the taeh3 tiny autoencoder.

    Returns images as `[N, H, W, C]` in [0, 1] with the same temporal frame count and
    spatial size as `latent_to_images`, so downstream chunker consumers behave the same.
    Returns None (fallback to `latent_to_images`) when taeh3 is unavailable or the decode
    fails for any reason.
    """
    if mode != "minimax-h3":
        return None
    decoder = _get_taeh3_decoder()
    if decoder is None:
        return None
    video = _extract_video(latent)
    if video is None:
        return None
    if decoder[1].weight.shape[1] != video.shape[1]:
        return None

    try:
        b, c, t, h, w = video.shape
        if t == 0:
            return None
        frames = []
        for i in range(t):
            if max_tokens is not None and i >= max_tokens:
                break
            with torch.no_grad():
                out = decoder(video[:, :, i].to(decoder[1].weight.device, decoder[1].weight.dtype))
            frames.append(out[0].movedim(0, -1).to(device=video.device, dtype=video.dtype).clamp(0, 1))
        rgb = torch.stack(frames, dim=0)

        target_frames = b * _frames_from_tokens(t, mode)
        if rgb.shape[0] != target_frames:
            src = rgb.shape[0]
            if target_frames == 1:
                idx = torch.zeros(1, dtype=torch.long, device=rgb.device)
            else:
                idx = (torch.arange(target_frames, device=rgb.device) * (src - 1) / (target_frames - 1)).round().long()
                if int(idx.max().item()) >= src:
                    idx = idx.clamp(max=src - 1)
            rgb = rgb[idx]

        target_w = w * 16
        target_h = h * 16
        if rgb.shape[2] != target_w or rgb.shape[1] != target_h:
            from .utils_tensor import resize_image
            rgb = resize_image(rgb, target_w, target_h)

        return rgb
    except Exception:
        return None