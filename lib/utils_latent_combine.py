from typing import List, Optional, Tuple

import safetensors.torch
import torch


def _load_latent_file(path: str):
    with safetensors.safe_open(path, framework="pt") as f:
        latent_type = f.metadata().get("type", "standard")
        latent_keys = [k for k in f.keys() if k.startswith("latent_")]
        if latent_keys:
            latent_keys.sort(key=lambda x: int(x.split("_")[1]))
            return [f.get_tensor(k) for k in latent_keys], latent_type, True
        return [f.get_tensor("latent")], latent_type, False


def _trim_tensor(t: torch.Tensor, video_overlap: int, audio_overlap: int, nested: bool, index: int) -> torch.Tensor:
    if nested and index > 0 and t.dim() == 4:
        # audio stream [B, C, 2, T]
        return t[:, :, :, audio_overlap:] if audio_overlap > 0 else t
    if t.dim() == 5:
        return t[:, :, video_overlap:, :, :] if video_overlap > 0 else t
    if t.dim() == 4 and video_overlap > 0:
        return t[video_overlap:]
    return t


def concat_chunk_latents(paths: List[str],
                         video_overlaps: Optional[List[int]] = None,
                         audio_overlaps: Optional[List[int]] = None) -> Tuple[torch.Tensor, str]:
    if not paths:
        return None, "standard"
    video_overlaps = video_overlaps or []
    audio_overlaps = audio_overlaps or []

    combined = None
    nested = False
    latent_type = "standard"

    for i, path in enumerate(paths):
        tensors, typ, is_nested = _load_latent_file(path)
        if i == 0:
            latent_type = typ
            nested = is_nested
        video_overlap = video_overlaps[i] if i < len(video_overlaps) else 0
        audio_overlap = audio_overlaps[i] if i < len(audio_overlaps) else 0
        if i == 0:
            video_overlap = 0
            audio_overlap = 0

        for j, t in enumerate(tensors):
            t = _trim_tensor(t, video_overlap, audio_overlap, nested, j)
            if combined is None:
                combined = [t]
            elif j >= len(combined):
                combined.append(t)
            else:
                if nested and j == 1:
                    dim = 3
                else:
                    dim = 2 if combined[j].dim() == 5 else 0
                combined[j] = torch.cat([combined[j], t], dim=dim)

    if nested:
        from comfy.nested_tensor import NestedTensor
        return NestedTensor([t for t in combined if t is not None]), latent_type
    return combined[0], latent_type