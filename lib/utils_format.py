from typing import Optional
import torch


def format_images(images: Optional[torch.Tensor]) -> str:
    return len(images) if images is not None else "0"


def format_masks(masks: Optional[torch.Tensor]) -> str:
    return len(masks) if masks is not None else "0"


def format_audio(audio: Optional[dict]) -> str:
    if audio is None:
        return "0s"
    return f"{audio['waveform'].shape[2] / audio['sample_rate']:.3f}s"


def format_fps(fps: Optional[float]) -> str:
    if fps is None:
        return "0"
    return f"{fps:.2f}"


def format_milliseconds(ms: int) -> str:
    if ms == 0:
        return "0"
    divisors = [1, 1000, 60, 60, 24, 7, 4, 13, 10, 10, 10]
    units = ['ms', 's', 'm', 'h', 'd', 'w', 'mo', 'y', 'dec', 'c', 'mi']
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
    return ''.join(out[0:2])
