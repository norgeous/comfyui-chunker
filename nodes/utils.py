from PIL import Image
import torch
import numpy as np
# import math
# from comfy.utils import common_upscale
from .textOverlay import batch_draw_text

def log(*args):
    print(f"\U0001F36B  Chunker:", *args)

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def panelImage(w, h, r=255, g=255, b=255):
    return pil2tensor(Image.new('RGB', (w, h), (r, g, b)))

def panelMask(w, h, v=255):
    return pil2tensor(Image.new('RGB', (w, h), (v, v, v)).convert('L'))

def mask_to_image(mask):
    result = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    return result

# def slice(thing, start=None, end=None):
#     if thing is None: return []
#     sliced = thing[start:end]
#     if len(sliced) == 0: return []
#     return [sliced]

# def len2(thing):
#     count = 0
#     for item in thing:
#         count += len(item)
#     return count

# def kijaiWanResizeCalc(image, generation_width, generation_height, aspect_ratio):
#     VAE_STRIDE = (4, 8, 8)
#     PATCH_SIZE = (1, 2, 2)
#     H, W = image.shape[1], image.shape[2]
#     max_area = generation_width * generation_height
#     crop = "disabled"
#     if aspect_ratio == "keep_input":
#         aspect_ratio = H / W
#     elif aspect_ratio == "stretch_to_new" or aspect_ratio == "crop_to_new":
#         aspect_ratio = generation_height / generation_width
#         if aspect_ratio == "crop_to_new":
#             crop = "center"
#     lat_h = round(
#     np.sqrt(max_area * aspect_ratio) // VAE_STRIDE[1] //
#     PATCH_SIZE[1] * PATCH_SIZE[1])
#     lat_w = round(
#         np.sqrt(max_area / aspect_ratio) // VAE_STRIDE[2] //
#         PATCH_SIZE[2] * PATCH_SIZE[2])
#     h = lat_h * VAE_STRIDE[1]
#     w = lat_w * VAE_STRIDE[2]
#     return (w, h, crop)

# def resizeImage(image, width, height, aspect_ratio):
#     if image is None: return None
#     w, h, crop = kijaiWanResizeCalc(image, width, height, aspect_ratio)
#     if image.shape[1] == h and image.shape[2] == w: return image
#     resized_image = common_upscale(image.movedim(-1, 1), w, h, "lanczos", crop).movedim(1, -1)
#     return resized_image

# def resizeMask(mask, width, height, aspect_ratio):
#     if mask is None: return None
#     w, h, crop = kijaiWanResizeCalc(mask, width, height, aspect_ratio)
#     if mask.shape[1] == h and mask.shape[2] == w: return mask
#     resized_mask = common_upscale(mask.unsqueeze(1).repeat(1, 3, 1, 1), w, h, "lanczos", crop).movedim(1,-1)[:, :, :, 0]
#     return resized_mask





def frameIndexInfo(i, previous_count, chunk_index, chunk_count, total, overlap):
    chunk = chunk_index + 1
    is_overlap = True if chunk_index > 0 and i < overlap else False
    return (
        f"{str(previous_count + i + 1).zfill(len(str(total)))} / {total}", # frame_label
        f"{str(chunk).zfill(len(str(chunk_count)))} of {chunk_count}", # chunk_label
        is_overlap,
        f"chunks {chunk - 1} + {chunk}", # overlap_label
    )

def getOverlayConfigs(i, previous_count, chunk_index, chunk_count, total, w, h, length, overlap):
    frame_label, chunk_label, is_overlap, overlap_label = frameIndexInfo(i, previous_count, chunk_index, chunk_count, total, overlap)
    configs = []
    configs.append(
        {
            "text": f"{frame_label}\n{chunk_label}",
            "vertical_alignment": "top",
            "horizontal_alignment": "right",
        },
    )
    configs.append(
        {
            "text": f"size: {w} x {h}\nchunk_length: {length}\nchunk_overlap: {overlap}",
            "font_size": 12,
            "vertical_alignment": "bottom",
            "horizontal_alignment": "right",
        },
    )
    if is_overlap:
        configs.append(
            {
                "text": "OVERLAP",
                "font_size": 24,
                "fill_color_hex": "#FF0000",
                "stroke_color_hex": "#FFFFFF",
                "vertical_alignment": "top",
                "horizontal_alignment": "left",
            },
        )
        configs.append(
            {
                "text": overlap_label,
                "font_size": 14,
                "fill_color_hex": "#FF0000",
                "stroke_color_hex": "#FFFFFF",
                "vertical_alignment": "top",
                "horizontal_alignment": "left",
                "y_shift": 24 + 4,
            },
        )
    return configs

def overlay_debug(images, previous_count, chunk_index, chunk_count, chunk_length, chunk_overlap, total_length):
    w = images.shape[2]
    h = images.shape[1]
    images = batch_draw_text(
        images,
        [getOverlayConfigs(i, previous_count, chunk_index, chunk_count, total_length, w, h, chunk_length, chunk_overlap) for i in range(0, len(images))],
    )
    return images
