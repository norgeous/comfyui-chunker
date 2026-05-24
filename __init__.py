import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

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


