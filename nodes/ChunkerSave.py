from ..lib.utils_av import profile_names, alpha_modes, save
from ..lib.utils_format import format_images, format_masks, format_audio, format_fps

class ChunkerSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("FLOAT", {"default": 30}),
                "profile": (profile_names, {"default": profile_names[0]}),
                "alpha_mode": (alpha_modes, {"default": alpha_modes[0]}),
                "path": ("STRING", {"default": "video/chunker/save"}),
            },
            "optional": {
                "images": ("IMAGE", {"default": None}),
                "masks": ("MASK", {"default": None}),
                "audio": ("AUDIO", {"default": None}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    OUTPUT_NODE = True

    def execute(self, fps, profile, alpha_mode, path, images=None, masks=None, audio=None):
        out_path, frontend_data = save(
            images=images,
            masks=masks,
            audio=audio,
            fps=fps,
            profile=profile,
            alpha_mode=alpha_mode,
            filename_prefix=path,
        )

        ui_values = {
            "input_label_values": {
                "images": format_images(images),
                "masks": format_masks(masks),
                "audio": format_audio(audio),
                "fps": format_fps(fps),
            },
            "video_path": frontend_data,
        }

        return {
            "ui": {"values": [ui_values]},
            "result": (
                out_path,
            ),
        }
