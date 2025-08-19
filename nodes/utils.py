from PIL import Image
import torch
import numpy as np
from comfy.utils import common_upscale

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def panelImage(w, h, r=255, g=255, b=255):
    return pil2tensor(Image.new('RGB', (w, h), (r, g, b)))

def panelMask(w, h, v=255):
    return pil2tensor(Image.new('RGB', (w, h), (v, v, v)).convert('L'))

def slice(thing, start=None, end=None):
    if thing is None: return []
    sliced = thing[start:end]
    if len(sliced) == 0: return []
    return [sliced]

def len2(thing):
    count = 0
    for item in thing:
        count += len(item)
    return count

def kijaiWanResizeCalc(image, generation_width, generation_height, aspect_ratio):
    VAE_STRIDE = (4, 8, 8)
    PATCH_SIZE = (1, 2, 2)
    H, W = image.shape[1], image.shape[2]
    max_area = generation_width * generation_height
    crop = "disabled"
    if aspect_ratio == "keep_input":
        aspect_ratio = H / W
    elif aspect_ratio == "stretch_to_new" or aspect_ratio == "crop_to_new":
        aspect_ratio = generation_height / generation_width
        if aspect_ratio == "crop_to_new":
            crop = "center"
    lat_h = round(
    np.sqrt(max_area * aspect_ratio) // VAE_STRIDE[1] //
    PATCH_SIZE[1] * PATCH_SIZE[1])
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // VAE_STRIDE[2] //
        PATCH_SIZE[2] * PATCH_SIZE[2])
    h = lat_h * VAE_STRIDE[1]
    w = lat_w * VAE_STRIDE[2]
    return (w, h, crop)

def resizeImage(image, width, height, aspect_ratio):
    if image is None: return None
    w, h, crop = kijaiWanResizeCalc(image, width, height, aspect_ratio)
    if image.shape[1] == h and image.shape[2] == w: return image
    resized_image = common_upscale(image.movedim(-1, 1), w, h, "lanczos", crop).movedim(1, -1)
    return resized_image

def resizeMask(mask, width, height, aspect_ratio):
    if mask is None: return None
    w, h, crop = kijaiWanResizeCalc(mask, width, height, aspect_ratio)
    if mask.shape[1] == h and mask.shape[2] == w: return mask
    resized_mask = common_upscale(mask.unsqueeze(1).repeat(1, 3, 1, 1), w, h, "lanczos", crop).movedim(1,-1)[:, :, :, 0]
    return resized_mask
