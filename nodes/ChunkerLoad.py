import folder_paths
from ..lib.utils_comfy import get_input_filenames
from ..lib.utils_av import load, alpha_modes
from ..lib.utils_format import format_audio, format_fps

class ChunkerLoad:
    @classmethod
    def INPUT_TYPES(cls):
        files = ["None", *get_input_filenames()]
        return {
            "required": {
                "alpha_mode": (alpha_modes, {"default": alpha_modes[0]}),
                "path": (files, {"default": "None", "tooltip": "Path to load"}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, path):
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
    DESCRIPTION = "ChunkerLoad"

    def execute(self, alpha_mode, path="None"):
        full_path = folder_paths.get_annotated_filepath(path)
        out_images, out_masks, out_audio, fps = load(path=full_path, alpha_mode=alpha_mode)

        ui_values = {
            "output_label_values": {
                "images": len(out_images) if out_images is not None else 0,
                "masks": len(out_masks) if out_masks is not None else 0,
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
