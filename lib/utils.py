import os
import folder_paths
from PIL import Image
import torch
import numpy as np
from comfy.utils import common_upscale
from functools import reduce
import math

def log(*args, **kwargs):
    print(f"\U0001F36B  Chunker:", *args, **kwargs)

def count(list):
    if len(list) == 0: return 0
    if len(list) == 1: return len(list[0])
    return reduce(lambda acc, item: acc + len(item), [0, *list])

# def pil2tensor(image):
#     return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    img_array = image.squeeze(0).cpu().numpy() * 255.0
    img_pil = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    return img_pil

# def panel_image(w, h, r=255, g=255, b=255):
#     return pil2tensor(Image.new('RGB', (w, h), (r, g, b)))

# def panel_mask(w, h, v=255):
#     return pil2tensor(Image.new('RGB', (w, h), (v, v, v)).convert('L'))

def mask_to_image(mask):
    if mask is None: return None;
    image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    return image

def image_to_mask(image):
    if image is None: return None;
    mask = image[:, :, :, 0] # keep first 3 dimensions, choose index 0 (red) for new channel
    return mask

def resize_image(image, width, height):
    if image is None: return None
    if image.shape[1] == height and image.shape[2] == width: return image
    resized_image = common_upscale(image.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
    return resized_image

def resize_mask(mask, width, height):
    image = mask_to_image(mask)
    resized_image = resize_image(image, width, height)
    return image_to_mask(resized_image)

def simple_blend(image1, image2, blend_factor=0.5):
    blended_image = image1 * (1 - blend_factor) + image2 * blend_factor
    blended_image = torch.clamp(blended_image, 0, 1)
    return blended_image

def force_wan_length(value):
    return (math.ceil((value - 1) / 4) * 4) + 1

def fix_total_length(total_length, chunk_length=49, chunk_overlap=2):
    if total_length <= chunk_length: return force_wan_length(total_length)
    adjusted_chunk_length = chunk_length - chunk_overlap
    full_length_chunk_count = (total_length) // adjusted_chunk_length
    final_chunk_length = (total_length) % adjusted_chunk_length
    corrected_final_chunk_length = force_wan_length(final_chunk_length)
    return (full_length_chunk_count * adjusted_chunk_length) + corrected_final_chunk_length

def get_this_chunk_length(index, chunk_length, chunk_overlap, total_length):
    adjusted_chunk_length = chunk_length - chunk_overlap
    full_length_chunk_count = (total_length) // adjusted_chunk_length
    if index < full_length_chunk_count: return chunk_length
    return total_length - (adjusted_chunk_length * full_length_chunk_count)

def get_input_filenames():
    files = []
    input_dir = folder_paths.get_input_directory()
    for f in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, f)):
            file_parts = f.split('.')
            if len(file_parts) > 1 and (file_parts[-1].lower() in ["mp4", "mov", "png", "jpeg", "jpg"]):
                files.append(f)
    return files
