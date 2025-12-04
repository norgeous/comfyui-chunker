from .lib.endpoints import * # importing this registers endpoint(s)

from .nodes.ChunkerMediaLoader import ChunkerMediaLoader
from .nodes.Chunker import Chunker
from .nodes.ChunkerVACEToFirstLast import ChunkerVACEToFirstLast
from .nodes.ChunkerCombine import ChunkerCombine


NODE_CONFIG = {
    "ChunkerMediaLoader": {"class": ChunkerMediaLoader, "name": "\U0001F36B Media Loader"},
    "Chunker": {"class": Chunker, "name": "\U0001F36B Chunker"},
    "ChunkerVACEToFirstLast": {"class": ChunkerVACEToFirstLast, "name": "\U0001F36B VACE To First Last"},
    "ChunkerCombine": {"class": ChunkerCombine, "name": "\U0001F36B Combine"},
}

NODE_CLASS_MAPPINGS = {
    "ChunkerMediaLoader": ChunkerMediaLoader,
    "Chunker": Chunker,
    "ChunkerVACEToFirstLast": ChunkerVACEToFirstLast,
    "ChunkerCombine": ChunkerCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChunkerMediaLoader": "\U0001F36B Media Loader",
    "Chunker": "\U0001F36B Chunker",
    "ChunkerVACEToFirstLast": "\U0001F36B VACE To First Last",
    "ChunkerCombine": "\U0001F36B Combine",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
