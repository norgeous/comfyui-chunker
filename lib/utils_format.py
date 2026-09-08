from typing import Optional
import torch


def format_images(images: Optional[torch.Tensor]) -> str:
    return len(images) if images is not None else "∅"


def format_masks(masks: Optional[torch.Tensor]) -> str:
    return len(masks) if masks is not None else "∅"


def format_audio(audio: Optional[dict]) -> str:
    if audio is None:
        return "∅"
    duration = audio["waveform"].shape[2] / audio["sample_rate"]
    sample_rate_k = audio["sample_rate"] / 1000.0
    dur_str = f"{duration:.3f}" if duration % 1 else f"{duration:.0f}"
    sr_str = f"{sample_rate_k:.1f}" if sample_rate_k % 1 else f"{sample_rate_k:.0f}"
    channels = audio["waveform"].shape[1] if audio["waveform"].dim() > 1 else 1
    icon = "\U0001F4FE" if channels > 1 else "\U0001F56C"
    return f"{dur_str}s ({sr_str}㎑ {icon})"


def format_fps(fps: Optional[float]) -> str:
    if fps is None:
        return "∅"
    return f"{fps:.2f}"


def format_boolean(value: bool) -> str:
    return "\u2705" if value else "\u274C"


def format_video(video) -> str:
    if video is None:
        return "∅"
    duration = video.get_duration()
    return f"{duration:.3f}s" if duration % 1 else f"{duration:.0f}s"


def format_milliseconds(ms: int) -> str:
    if ms == 0:
        return "0"
    divisors = [1, 1000, 60, 60, 24, 7, 4, 13, 10, 10, 10]
    units = ["ms", "s", "m", "h", "d", "w", "mo", "y", "dec", "c", "mi"]
    results = []
    quotient = ms
    for i in range(1, len(divisors)):
        results.append(quotient % divisors[i])
        quotient //= divisors[i]
    results.append(quotient)
    rresults = list(reversed(results))
    runits = list(reversed(units))
    first = next(i for i, v in enumerate(rresults) if v > 0)
    last = len(results) - next(i for i, v in enumerate(results) if v > 0)
    out = [f"{rresults[i]}{runits[i]}" for i in range(first, last)]
    return "".join(out[0:2])


def format_latent(latent: Optional[dict]) -> str:
    if latent is None:
        return "∅"
    if isinstance(latent, dict):
        latent_tensor = latent.get("samples")
    else:
        latent_tensor = latent
    
    if latent_tensor is None:
        return "∅"
    
    if hasattr(latent_tensor, "tensors"):  # NestedTensor
        parts = []
        for t in latent_tensor.tensors:
            if t.dim() == 5:  # Video: [B, C, T, H, W]
                parts.append(str(t.shape[2]))
            elif t.dim() == 4 and t.shape[2] == 2:  # Audio: [B, C, 2, T]
                parts.append(str(t.shape[3]))
        return ", ".join(parts) if parts else "∅"
    
    if latent_tensor.dim() == 5:  # Video: [B, C, T, H, W]
        return str(latent_tensor.shape[2])
    elif latent_tensor.dim() == 4 and latent_tensor.shape[2] == 2:  # Audio: [B, C, 2, T]
        return str(latent_tensor.shape[3])
    
    return "∅"
