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
    "Chunker": "\U0001F36B Chunker",
    "ChunkerCombine": "\U0001F36B Combine",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
