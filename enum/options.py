from enum import Enum

class OverlapBlendModes(Enum):
    # preserve_this_chunk = 1
    # preserve_previous_chunk = 2
    # linear_blend = 3 # TODO
    # ease_in_out = 4 # TODO
    older_only = 1
    linear_blend = 2 # TODO
    ease_in_out = 3 # TODO
    newer_only = 4

class DivideMode(Enum):
    default = 1
    wan = 2
    wan_vace = 3
    ltx2 = 4
