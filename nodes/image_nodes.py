from PIL import Image
import torch
import numpy as np

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class Chunker:
    RETURN_TYPES = ("CHUNK_INFO", "IMAGE", "MASK", "INT",)
    RETURN_NAMES = ("chunk_info", "control_video", "control_masks", "length",)
    OUTPUT_TOOLTIPS = (
        "chunk_info",
        "Chunk of control_video for WanVaceToVideo",
        "Chunk of control_masks for WanVaceToVideo",
        "The length of this chunk",
    )
    FUNCTION = "chunkfrombatch"
    CATEGORY = "Chunker"
    DESCRIPTION = "Chunker for WanVaceToVideo for overlapping video extension, inpainting and outpainting"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "mode": (["extend","inpaint"], {"default": "extend", "tooltip": "Extend: The first chunk starts from the last image in images.\nInpaint: The first chunk starts from the first image in images."}),
                 "index": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, "tooltip": "Current chunk index"}),
                 "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4, "tooltip": "Count of frames in this chunk"}),
                 "overlap": ("INT", {"default": 2, "min": 0, "max": 4096, "step": 1, "tooltip": "Count of frames to overlap between chunks"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Single image or image sequence (video) to be chunked"}),
                "control_video": ("IMAGE", {"tooltip": "control_video to be chunked"}),
                "control_masks": ("MASK", {"tooltip": "control_masks to be chunked"}),
            }
        }

    def chunkfrombatch(self, mode, index, length, overlap, images=None, control_video=None, control_masks=None):
        extend_from_images = mode == "extend"

        images_before = None #torch.empty(0, 1)
        images_overlap = None
        # images_chunk = None
        images_after = None
        control_video_chunk = None

        adjusted_overlap = 0 if index == 0 else overlap # exclude overlap in first chunk
        adjusted_length = length - adjusted_overlap

        w = images.shape[2] if images is not None else 500
        h = images.shape[1] if images is not None else 500

        overlap_start = index * adjusted_length
        chunk_start = overlap_start + adjusted_overlap
        after_start = chunk_start + adjusted_length

        # copy last image if not enough images to fill length
        if images is not None and len(images) < after_start:
            images = torch.cat((images, images[-1].repeat(after_start - len(images), 1, 1, 1)), dim=0)
        if control_video is not None and len(control_video) < after_start:
            control_video = torch.cat((control_video, control_video[-1].repeat(after_start - len(control_video), 1, 1, 1)), dim=0)







        # select chunk of images if provided
        if images is not None:
            image_count = len(images)
            images_before = images[0:overlap_start]
            images_overlap = images[overlap_start:chunk_start]
            # images_chunk = images[chunk_start:after_start]
            images_after = images[after_start:image_count]






        grey_panel  = pil2tensor(Image.new('RGB', (w, h), (128, 128, 128)))

        # create control_video
        control_video_chunk = []
        if images is not None:
            control_video_chunk.extend([images_overlap])
        if control_video is None:
            if images is not None and extend_from_images and index == 0:
                #print("start with real image")
                control_video_chunk.append(images[0:1])
                control_video_chunk.extend([grey_panel] * (adjusted_length - 1))
            else:
                #print('all grey images')
                control_video_chunk.extend([grey_panel] * adjusted_length)
        else:
            control_video_chunk.extend([control_video[chunk_start:after_start]])
        #print('images',images)
        #print('cvlen',len(control_video_chunk))
        #print('cv',control_video_chunk)
        control_video_torch = torch.cat(control_video_chunk, dim=0)





        black_panel = pil2tensor(Image.new('RGB', (w, h), (0,   0,   0  )).convert('L'))
        white_panel = pil2tensor(Image.new('RGB', (w, h), (255, 255, 255)).convert('L'))

        # create control_masks
        control_masks_chunk = []
        if images is not None:
            control_masks_chunk.extend([black_panel] * (1 if extend_from_images and index == 0 else adjusted_overlap))
        if control_masks is None:
            if images is not None and extend_from_images and index == 0:
                control_masks_chunk.extend([white_panel] * (adjusted_length - 1))
            else:
                control_masks_chunk.extend([white_panel] * adjusted_length)
        else:
            control_masks_chunk.extend([control_masks[overlap_start:after_start]])
        control_masks_torch = torch.cat(control_masks_chunk, dim=0)

        chunk_info = (
            images_before,
            images_after,
        )

        #print("chunk_info send", chunk_info)

        return (
            chunk_info,
            control_video_torch,
            control_masks_torch,
            length,
        )
























class ImageBatchMulti:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "Chunker"
    DESCRIPTION = "Chunker Combine"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "chunk_info": ("CHUNK_INFO", {"forceInput": True}),
                "images": ("IMAGE", ),
            },
    }

    def combine(self, chunk_info, images):
        output = []
        print("chunk_info get", chunk_info)
        if chunk_info[0] is not None:
            output.extend([chunk_info[0]])
        if images is not None:
            output.extend([images])
        if chunk_info[1] is not None:
            output.extend([chunk_info[1]])
        output_torch = torch.cat(output, dim=0)
        return (output_torch,)

