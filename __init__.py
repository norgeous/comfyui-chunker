from .nodes.endpoints import * # register endpoint?
from .nodes.nodes import Chunker, ChunkerCombine, ChunkerVACEToFirstLast, ChunkerChunkConfig, ChunkerOutpaintConfig

NODE_CONFIG = {
    "ChunkerOutpaintConfig": {"class": ChunkerOutpaintConfig, "name": "\U0001F36B Outpaint Config"},
    "ChunkerChunkConfig": {"class": ChunkerChunkConfig, "name": "\U0001F36B Chunk Config"},
    "Chunker": {"class": Chunker, "name": "\U0001F36B Chunker"},
    "ChunkerVACEToFirstLast": {"class": ChunkerVACEToFirstLast, "name": "\U0001F36B VACE To First Last"},
    "ChunkerCombine": {"class": ChunkerCombine, "name": "\U0001F36B Combine"},
}

NODE_CLASS_MAPPINGS = {
    "ChunkerOutpaintConfig": ChunkerOutpaintConfig,
    "ChunkerChunkConfig": ChunkerChunkConfig,
    "Chunker": Chunker,
    "ChunkerVACEToFirstLast": ChunkerVACEToFirstLast,
    "ChunkerCombine": ChunkerCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChunkerOutpaintConfig": "\U0001F36B Outpaint Config",
    "ChunkerChunkConfig": "\U0001F36B Chunk Config",
    "Chunker": "\U0001F36B Chunker",
    "ChunkerVACEToFirstLast": "\U0001F36B VACE To First Last",
    "ChunkerCombine": "\U0001F36B Combine",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
