from ..lib.av.prores_flac_saver import save_prores_with_alpha
from ..lib.av.save import save_prores_mkv_with_alpha_and_audio

class ChunkerSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"default": None}),
                "masks": ("MASK", {"default": None}),
                "audio": ("AUDIO", {"default": None}),
                "fps": ("FLOAT", {"default": 30}),
                "filename_prefix": ("STRING", {"default": "chunker_save"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "execute"
    CATEGORY = "Chunker"

    def execute(self, images=None, masks=None, audio=None, fps=30.0, filename_prefix="chunker_save"):
        save_prores_mkv_with_alpha_and_audio(path="test.mkv", images=images, masks=masks, audio=audio, fps=fps)
        return ("test.mkv",)
