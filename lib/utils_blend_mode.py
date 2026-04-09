from enum import Enum
from typing import Callable


class OverlapBlendMode(Enum):
    LINEAR_BLEND = "linear_blend"
    EASE_IN_OUT = "ease_in_out"
    NEWER = "newer_only"
    OLDER = "older_only"


BlendHandler = Callable[[float], float]

handlers: dict[OverlapBlendMode, BlendHandler] = {
    OverlapBlendMode.LINEAR_BLEND: lambda p: p,
    OverlapBlendMode.EASE_IN_OUT: lambda p: (p * p * (3 - 2 * p)),
    OverlapBlendMode.NEWER: lambda p: float(p != 0),
    OverlapBlendMode.OLDER: lambda p: float(p == 1),
}


def get_blend_factor(mode: OverlapBlendMode, percent: float) -> float:
    return 1 - handlers[mode](percent)
