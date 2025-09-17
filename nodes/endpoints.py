import os
import folder_paths
import torch
import math
from comfy_execution.graph_utils import GraphBuilder
from .utils import log, tensor2pil, panelImage, panelMask, mask_to_image, image_to_mask, resize_image, resize_mask, create_preview_video, get_input_filenames
from .repeatNodes import comfyuiRepeatNodes, getNodeIdsByType
from .loader import awesome_loader, quick_combine, save_video
from .loadAudio import load_audio

# Add custom API routes, using router
from aiohttp import web
from server import PromptServer
from urllib.parse import unquote
from PIL import Image

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
