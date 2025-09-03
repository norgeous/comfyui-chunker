#from .nodes.image_nodes import *
from .nodes.nodes import *

NODE_CONFIG = {
    "ChunkerConfig": {"class": ChunkerConfig, "name": "\U0001F36B Config"},
    "ChunkerResequencer": {"class": ChunkerResequencer, "name": "\U0001F36B Resequencer"},
    "Chunker": {"class": Chunker, "name": "\U0001F36B Chunker"},
    "ChunkerCombine": {"class": ChunkerCombine, "name": "\U0001F36B Combine"},
}

NODE_CLASS_MAPPINGS = {
    "ChunkerConfig": ChunkerConfig,
    "ChunkerResequencer": ChunkerResequencer,
    "Chunker": Chunker,
    "ChunkerCombine": ChunkerCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChunkerConfig": "\U0001F36B Config",
    "ChunkerResequencer": "\U0001F36B Resequencer",
    "Chunker": "\U0001F36B Chunker",
    "ChunkerCombine": "\U0001F36B Combine",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
