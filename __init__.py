from .lib.endpoints import * # importing this registers endpoint(s)

from .nodes.ChunkerMediaLoader import ChunkerMediaLoader
from .nodes.ChunkerDivide import ChunkerDivide
from .nodes.ChunkerVACEToFirstLast import ChunkerVACEToFirstLast
from .nodes.ChunkerCombine import ChunkerCombine


NODE_CONFIG = {
    "ChunkerMediaLoader": {"class": ChunkerMediaLoader, "name": "\U0001F36B Media Loader"},
    "ChunkerDivide": {"class": ChunkerDivide, "name": "\U0001F36B Divide"},
    "ChunkerVACEToFirstLast": {"class": ChunkerVACEToFirstLast, "name": "\U0001F36B VACE To First Last"},
    "ChunkerCombine": {"class": ChunkerCombine, "name": "\U0001F36B Combine"},
}

NODE_CLASS_MAPPINGS = {
    "ChunkerMediaLoader": ChunkerMediaLoader,
    "ChunkerDivide": ChunkerDivide,
    "ChunkerVACEToFirstLast": ChunkerVACEToFirstLast,
    "ChunkerCombine": ChunkerCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChunkerMediaLoader": "\U0001F36B Media Loader",
    "ChunkerDivide": "\U0001F36B Divide",
    "ChunkerVACEToFirstLast": "\U0001F36B VACE To First Last",
    "ChunkerCombine": "\U0001F36B Combine",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
