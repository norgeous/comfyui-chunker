from .nodes.endpoints import * # register endpoint?
from .nodes.nodes import Chunker, ChunkerCombine, ChunkerVACEToFirstLast, ChunkerChunkConfig

NODE_CONFIG = {
    "ChunkerChunkConfig": {"class": Chunker, "name": "\U0001F36B Config"},
    "Chunker": {"class": Chunker, "name": "\U0001F36B Chunker"},
    "ChunkerVACEToFirstLast": {"class": ChunkerVACEToFirstLast, "name": "\U0001F36B VACE To First Last"},
    "ChunkerCombine": {"class": ChunkerCombine, "name": "\U0001F36B Combine"},
}

NODE_CLASS_MAPPINGS = {
    "ChunkerChunkConfig": ChunkerChunkConfig,
    "Chunker": Chunker,
    "ChunkerVACEToFirstLast": ChunkerVACEToFirstLast,
    "ChunkerCombine": ChunkerCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChunkerChunkConfig": "\U0001F36B Config",
    "Chunker": "\U0001F36B Chunker",
    "ChunkerVACEToFirstLast": "\U0001F36B VACE To First Last",
    "ChunkerCombine": "\U0001F36B Combine",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
