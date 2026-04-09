from av import Packet
from .utils_blend_mode import get_blend_factor, OverlapBlendMode

def blend_videos(mode: OverlapBlendMode, all_packets: list[list[Packet]], overlap_count=10) -> list[Packet]:
    for packets in all_packets:
        is_first = True
        is_last = True
    # get_blend_factor(mode)