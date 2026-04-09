from enum import Enum
from typing import Callable


class OverlapBlendMode(Enum):
    LINEAR_BLEND = "linear_blend"
    EASE_IN_OUT = "ease_in_out"
    NEWER = "newer"
    OLDER = "older"


BlendHandler = Callable[[float], float]

handlers: dict[OverlapBlendMode, BlendHandler] = {
    OverlapBlendMode.LINEAR_BLEND: lambda p: p,
    OverlapBlendMode.EASE_IN_OUT: lambda p: (p * p * (3 - 2 * p)),
    OverlapBlendMode.NEWER: lambda p: 0.0 if p > 0 else 1.0,
    OverlapBlendMode.OLDER: lambda p: 1.0 if p < 1 else 0.0,
}


def get_blend_factor(mode: OverlapBlendMode, percent: float) -> float:
    return 1 - handlers[mode](percent)
