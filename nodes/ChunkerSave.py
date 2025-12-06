from ..lib.av.save import save

class ChunkerSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("FLOAT", {"default": 30}),
                "filename_prefix": ("STRING", {"default": "chunker_save"}),
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

    def execute(self, images=None, masks=None, audio=None, fps=30.0, filename_prefix="chunker_save"):
        path = save(images=images, masks=masks, audio=audio, fps=fps, filename_prefix=filename_prefix)
        return (path,)
