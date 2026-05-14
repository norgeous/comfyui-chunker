from comfy_api.latest import ComfyExtension, io

from .nodes.ChunkerDivide import ChunkerDivide
from .nodes.ChunkerCombine import ChunkerCombine


class Chunker(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ChunkerDivide,
            ChunkerCombine,
        ]

async def comfy_entrypoint() -> ComfyExtension:
    return Chunker()

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
