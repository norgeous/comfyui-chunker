from comfy_api.latest import ComfyExtension, io

from .lib import execution_monitor  # noqa: F401 - patches execution at import time
from .nodes.ChunkerRepeat import ChunkerRepeat
from .nodes.ChunkerCombine import ChunkerCombine
from .nodes.ChunkerData import ChunkerData


class Chunker(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ChunkerRepeat,
            ChunkerCombine,
            ChunkerData,
        ]


async def comfy_entrypoint() -> ComfyExtension:
    return Chunker()

WEB_DIRECTORY = "./js"
