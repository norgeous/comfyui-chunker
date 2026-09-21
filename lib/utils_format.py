from typing import Optional
import torch


def format_images(images: Optional[torch.Tensor]) -> str:
    if images is None:
        return "\u2205"
    return f"{images.shape[0]}\u00D7{images.shape[2]}\u00D7{images.shape[1]}"


def format_masks(masks: Optional[torch.Tensor]) -> str:
    if masks is None:
        return "\u2205"
    return f"{masks.shape[0]}\u00D7{masks.shape[2]}\u00D7{masks.shape[1]}"


def format_audio(audio: Optional[dict]) -> str:
    if audio is None or not isinstance(audio, dict):
        return "\u2205"
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if waveform is None or not sample_rate:
        return "\u2205"
    duration = waveform.shape[-1] / sample_rate
    sample_rate_k = sample_rate / 1000.0
    dur_str = f"{duration:.3f}" if duration % 1 else f"{duration:.0f}"
    sr_str = f"{sample_rate_k:.1f}" if sample_rate_k % 1 else f"{sample_rate_k:.0f}"
    channels = waveform.shape[-2] if waveform.dim() > 1 else 1
    icon = "\U0001F4FE" if channels > 1 else "\U0001F56C"
    return f"{dur_str}s {sr_str}\u3391 {icon} "


def format_fps(fps: Optional[float]) -> str:
    if fps is None:
        return "\u2205"
    return f"{fps:.2f}"


def format_boolean(value: bool) -> str:
    return "\u2705" if value else "\u274C"


def format_video(video) -> str:
    if video is None:
        return "\u2205"
    try:
        frame_count = video.get_frame_count()
        width, height = video.get_dimensions()
        fps = float(video.get_frame_rate())
    except (ValueError, ZeroDivisionError):
        return "\u2205"
    return f"{frame_count}\u00D7{width}\u00D7{height}@{fps:.2f}fps"


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
        return "\u2205"
    if isinstance(latent, dict):
        latent_tensor = latent.get("samples")
    else:
        latent_tensor = latent
    
    if latent_tensor is None:
        return "\u2205"
    
    suffix = " \U0001F17C" if isinstance(latent, dict) and "noise_mask" in latent else ""
    
    if hasattr(latent_tensor, "tensors"):  # NestedTensor
        parts = []
        for t in latent_tensor.tensors:
            if t.dim() == 5:  # Video: [B, C, T, H, W]
                parts.append(str(t.shape[2]) + "🎞")
            elif t.dim() == 4 and t.shape[2] == 2:  # Audio: [B, C, 2, T]
                parts.append(str(t.shape[3]) + "∿")
        return (", ".join(parts) if parts else "\u2205") + suffix
    
    if latent_tensor.dim() == 5:  # Video: [B, C, T, H, W]
        return str(latent_tensor.shape[2]) + suffix
    elif latent_tensor.dim() == 4 and latent_tensor.shape[2] == 2:  # Audio: [B, C, 2, T]
        return str(latent_tensor.shape[3]) + suffix
    
    return "\u2205" + suffix
