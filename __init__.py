#from .nodes.image_nodes import *
from .nodes.logic import *

NODE_CONFIG = {
    "Chunker": {"class": Chunker, "name": ""},
    "ChunkerCombine": {"class": ChunkerCombine, "name": ""},
}

NODE_CLASS_MAPPINGS = {
    "Chunker": Chunker,
    "ChunkerCombine": ChunkerCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Chunker": "Chunker",
    "ChunkerCombine": "ChunkerCombine",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
