from typing import Optional
import torch


def format_images(images: Optional[torch.Tensor]) -> str:
    return len(images) if images is not None else "0"


def format_masks(masks: Optional[torch.Tensor]) -> str:
    return len(masks) if masks is not None else "0"


def format_audio(audio: Optional[dict]) -> str:
    if audio is None:
        return "0"
    duration = audio["waveform"].shape[2] / audio["sample_rate"]
    sample_rate_k = audio["sample_rate"] / 1000.0
    dur_str = f"{duration:.1f}" if duration % 1 else f"{duration:.0f}"
    sr_str = f"{sample_rate_k:.1f}" if sample_rate_k % 1 else f"{sample_rate_k:.0f}"
    channels = audio["waveform"].shape[1] if audio["waveform"].dim() > 1 else 1
    icon = "\U0001F4FE" if channels > 1 else "\U0001F56C"
    return f"{dur_str}s ({sr_str}㎑ {icon})"


def format_fps(fps: Optional[float]) -> str:
    if fps is None:
        return "0"
    return f"{fps:.2f}"


def format_boolean(value: bool) -> str:
    return "\u2705" if value else "\u274C"


def format_video(video) -> str:
    if video is None:
        return "0"
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
