from comfy_api.latest import ComfyExtension, io

from .lib.endpoints import * # importing this registers endpoint(s)

from .nodes.ChunkerDivide import ChunkerDivide
from .nodes.ChunkerVACEToFirstLast import ChunkerVACEToFirstLast
from .nodes.ChunkerCombine import ChunkerCombine

from .nodes.ChunkerStorylines import ChunkerStorylines
from .nodes.ChunkerLoraSequencer import ChunkerLoraSequencer

# from .nodes.ChunkerLoad import ChunkerLoad
# from .nodes.ChunkerSave import ChunkerSave
# from .nodes.TensorDebug import TensorDebug

class Chunker(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ChunkerDivide,
            ChunkerCombine,
            # ChunkerStorylines,
            # ChunkerLoraSequencer,
        ]

async def comfy_entrypoint() -> ComfyExtension:
    return Chunker()

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
