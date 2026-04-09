from enum import Enum


class BlendMode(Enum):
    LINEAR_BLEND = "linear_blend"
    EASE_IN_OUT = "ease_in_out"
    NEWER = "newer"
    OLDER = "older"


handlers = {
    BlendMode.LINEAR_BLEND: lambda p: 1 - p,
    BlendMode.EASE_IN_OUT: lambda p: 1 - (p * p * (3 - 2 * p)),
    BlendMode.NEWER: lambda p: 1.0 if p == 0 else 0.0,
    BlendMode.OLDER: lambda p: 0.0 if p == 1.0 else 1.0,
}


def get_blend_factor(mode: BlendMode, percent: float) -> float:
    return 1 - handlers[mode](percent)
