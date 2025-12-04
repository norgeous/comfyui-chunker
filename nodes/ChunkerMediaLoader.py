from ..lib.utils import get_input_filenames
from ..lib.av.loader import media_loader
from ..lib.format_utils import format_audio, format_fps

class ChunkerMediaLoader:
    @classmethod
    def INPUT_TYPES(cls):
        files = ["None", *sorted(get_input_filenames())]
        return {
            "required": {
                "images": (files, {"default": "None", "tooltip": "Images"}),
                "masks": (files, {"default": "None", "tooltip": "Masks"}),
            },
            "optional": {
                "image": (files,),
                "image_paint": (files,),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, images, masks, image, image_paint):
        # YOLO, anything goes!
        return True

    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "masks", "audio", "fps")
    OUTPUT_TOOLTIPS = (
        "images",
        "masks",
        "audio",
        "fps",
    )
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    DESCRIPTION = "ChunkerMediaLoader"

    def execute(
        self,
        images,
        masks,
        image="None",
        image_paint="None",
    ):
        out_images, out_masks, out_audio, fps = media_loader(images, masks, image, image_paint)

        ui_values = {
            "output_label_values": {
                "images": len(out_images),
                "masks": len(out_masks),
                "audio": format_audio(out_audio),
                "fps": format_fps(fps),
            },
        }

        return {
            "ui": {"values": [ui_values]},
            "result": (
                out_images,
                out_masks,
                out_audio,
                fps,
            ),
        }
