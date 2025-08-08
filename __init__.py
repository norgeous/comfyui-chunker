from .nodes.image_nodes import *

NODE_CONFIG = {
    "Chunker": {"class": Chunker, "name": "Itterate video chunks"},
}

NODE_CLASS_MAPPINGS = { "Chunker": Chunker }

NODE_DISPLAY_NAME_MAPPINGS = { "Chunker": "Chunker" }

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
