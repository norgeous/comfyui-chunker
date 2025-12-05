import torch
from ..lib.utils import (
    log,
    mask_to_image,
    image_to_mask,
)
from ..lib.debug_overlay import create_preview_video
from ..lib.repeat_nodes import comfyui_repeat_nodes, get_node_ids_by_type
from ..lib.av.loader import awesome_loader, quick_combine, save_video, save_audio
from ..lib.av.load_audio import concat_audios

class ChunkerVACEToFirstLast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision": ("CLIP_VISION",),
                "images": ("IMAGE", {"tooltip": "VACE image sequence"}),
                "crop": (["none", "center"],)
            },
        }

    RETURN_TYPES = ("CLIP_VISION_OUTPUT", "CLIP_VISION_OUTPUT", "IMAGE", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("clip_vision_start_image", "clip_vision_end_image", "start_image", "end_image", "width", "height", "length")
    OUTPUT_TOOLTIPS = (
        "Start clip vison or None",
        "End clip vision or None",
        "Start image or None",
        "End image or None",
        "width",
        "height",
        "length",
    )
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    DESCRIPTION = "ChunkerVACEToFirstLast"

    def execute(self, clip_vision, images, crop):
        length = images.shape[0]
        h = images.shape[1]
        w = images.shape[2]

        clip_vision_start_image = None
        start_image = images[0].unsqueeze(0)
        if round(torch.mean(start_image).item(), 4) == round(127 / 255, 4): # detect grey panel start image
            start_image = None
        else:
            clip_vision_start_image = clip_vision.encode_image(start_image, crop=False if crop == "center" else True)

        clip_vision_end_image = None
        end_image = images[length - 1].unsqueeze(0)
        if round(torch.mean(end_image).item(), 4) == round(127 / 255, 4): # detect grey panel end image
            end_image = None
        else:
            clip_vision_end_image = clip_vision.encode_image(end_image, crop=False if crop == "center" else True)

        ui_values = {
            "input_label_values": {
                "images": len(images) if images is not None else 0,
            },
            "output_label_values": {
                "clip_vision_start_image": 1 if clip_vision_start_image is not None else 0,
                "clip_vision_end_image": 1 if clip_vision_end_image is not None else 0,
                "start_image": 1 if start_image is not None else 0,
                "end_image": 1 if end_image is not None else 0,
                "width": w,
                "height": h,
                "length": length,
            },
        }

        return {
            "ui": {"values": [ui_values]},
            "result": (
                clip_vision_start_image,
                clip_vision_end_image,
                start_image,
                end_image,
                w,
                h,
                length,
            ),
        }
