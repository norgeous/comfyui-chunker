from ..lib.utils_av import save

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
        path = save(
            images=images,
            masks=masks,
            audio=audio,
            fps=fps,
            filename_prefix="video/chunker/lossless_save",
        )
        return (path,)
