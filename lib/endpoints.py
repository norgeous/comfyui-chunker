import os
import folder_paths
from aiohttp import web
from server import PromptServer
from urllib.parse import unquote

from .utils import tensor2pil
from .av.loader import awesome_loader

# register /api/chunker/get-first-frame?filename=example.mp4
@PromptServer.instance.routes.get("/chunker/get-first-frame")
async def get_first_frame(request):
    if "filename" in request.query:
        filename = unquote(request.query["filename"])
        filepath = os.path.join(folder_paths.get_input_directory(), filename)

        # check input file exists
        if not os.path.isfile(filepath): return web.HTTPBadRequest()

        first_frames_dir = os.path.join(folder_paths.get_input_directory(), "first-frame")
        out_file = f"{os.path.basename(filename)}.png"
        out_path = os.path.join(first_frames_dir, out_file)
        frontend_data = {
            "type": "input",
            "filename": out_file,
            "subfolder": "first-frame",
        }

        # check if png already created
        if os.path.isfile(out_path): return web.json_response(frontend_data)

        # mkdir input/first-frame/
        if not os.path.isdir(first_frames_dir): os.mkdir(first_frames_dir)

        image = awesome_loader(filepath, 0, 1)[0]
        img = tensor2pil(image)
        img.save(out_path)
        return web.json_response(frontend_data)
    else:
        return web.HTTPBadRequest()
