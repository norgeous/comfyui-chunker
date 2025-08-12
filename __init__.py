from .nodes.image_nodes import *
from .nodes.logic import *

NODE_CONFIG = {
    "Chunker": {"class": Chunker, "name": "Itterate video chunks"},
    "ChunkerCombine": {"class": ImageBatchMulti, "name": ""},
    "ChunkerForLoopStart": {"class": forLoopStart, "name": ""},
    "ChunkerCombine2": {"class": ChunkerCombine2, "name": ""},
}

NODE_CLASS_MAPPINGS = {
    "Chunker": Chunker,
    "ChunkerCombine": ImageBatchMulti,
    "ChunkerForLoopStart": forLoopStart,
    "ChunkerCombine2": ChunkerCombine2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Chunker": "Chunker",
    "ChunkerCombine": "ChunkerCombine",
    "ChunkerForLoopStart": "ChunkerForLoopStart",
    "ChunkerCombine2": "ChunkerCombine2",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
