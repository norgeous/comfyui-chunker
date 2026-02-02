from enum import Enum

class OverlapBlendModes(Enum):
    preserve_this_chunk = 1
    preserve_previous_chunk = 2
    # linear_blend = 3 # TODO
    # ease_in_out = 4 # TODO

    # older_only
    # linear_blend
    # ease_in_out
    # newer_only