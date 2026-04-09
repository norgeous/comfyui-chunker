from enum import Enum
from typing import Callable


class OverlapBlendMode(Enum):
    LINEAR_BLEND = "linear_blend"
    EASE_IN_OUT = "ease_in_out"
    NEWER_ONLY = "newer_only"
    OLDER_ONLY = "older_only"


BlendHandler = Callable[[float], float]

handlers: dict[OverlapBlendMode, BlendHandler] = {
    OverlapBlendMode.LINEAR_BLEND: lambda p: p,
    OverlapBlendMode.EASE_IN_OUT: lambda p: (p * p * (3 - 2 * p)),
    OverlapBlendMode.NEWER_ONLY: lambda p: float(p != 0),
    OverlapBlendMode.OLDER_ONLY: lambda p: float(p == 1),
}


def get_blend_factor(mode: OverlapBlendMode, percent: float) -> float:
    return 1 - handlers[mode](percent)
