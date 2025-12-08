import folder_paths
from ..lib.utils import get_input_filenames
from ..lib.utils_av import load
from ..lib.utils_format import format_audio, format_fps

class ChunkerLoad:
    @classmethod
    def INPUT_TYPES(cls):
        files = ["None", *sorted(get_input_filenames())]
        return {
            "required": {
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

    def execute(
        self,
        path="None",
    ):
        # print('1', path)
        full_path = folder_paths.get_annotated_filepath(path)
        # print('2', full_path)
        out_images, out_masks, out_audio = load(path=full_path)
        # print('3')
        fps = 30

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
