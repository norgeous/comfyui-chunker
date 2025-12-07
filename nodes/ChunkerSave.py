from ..lib.utils_comfy import get_next_save_video_path
from ..lib.av.save import save

class ChunkerSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("FLOAT", {"default": 30}),
            },
            "optional": {
                "images": ("IMAGE", {"default": None}),
                "masks": ("MASK", {"default": None}),
                "audio": ("AUDIO", {"default": None}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "execute"
    CATEGORY = "Chunker"

    def execute(self, images=None, masks=None, audio=None, fps=30.0):
        filename_prefix = "video/chunker/tmp/chunk/lossless"
        full_path = get_next_save_video_path(filename_prefix)[0]
        path = save(images=images, masks=masks, audio=audio, fps=fps, path=full_path)
        return (path,)
