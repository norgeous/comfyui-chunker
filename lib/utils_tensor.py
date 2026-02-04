from PIL import Image
import torch
import numpy as np
from comfy.utils import common_upscale

# def pil2tensor(image):
#     return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    img_array = image.squeeze(0).cpu().numpy() * 255.0
    img_pil = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
    return img_pil

def monochrome_image(w, h, v=0.5):
    return torch.full((1, h, w, 3), v, dtype=torch.float32)

def monochrome_mask(w, h, v=1.0):
    return torch.full((1, h, w), v, dtype=torch.float32)

def mask_to_image(mask):
    if mask is None: return None
    image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    return image

def image_to_mask(image):
    if image is None: return None
    mask = image[:, :, :, 0] # keep first 3 dimensions, choose index 0 (red) for new channel
    return mask

def resize_image(image, width, height):
    if image is None: return None
    if image.shape[1] == height and image.shape[2] == width: return image
    resized_image = common_upscale(image.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
    return resized_image

def resize_mask(mask, width, height):
    imask = mask_to_image(mask)
    resized_imask = resize_image(imask, width, height)
    return image_to_mask(resized_imask)

def simple_blend(image1, image2, blend_factor=0.5):
    blended_image = image1 * (1 - blend_factor) + image2 * blend_factor
    blended_image = torch.clamp(blended_image, 0, 1)
    return blended_image

# https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/image_nodes.py#L1824
def temporal_blend(image1, image2, mode):
    blend_src = image1
    blend_dst = image2
    overlap = len(image1)

    if mode == "linear_blend":
        alpha = torch.linspace(0, 1, overlap + 2, device=blend_src.device, dtype=blend_src.dtype)[1:-1]
        alpha = alpha.view(-1, 1, 1, 1)  # Shape: [overlap, 1, 1, 1]
        blended_images = (1 - alpha) * blend_src + alpha * blend_dst

    elif mode == "filmic_crossfade":
        gamma = 2.2
        alpha = torch.linspace(0, 1, overlap + 2, device=blend_src.device, dtype=blend_src.dtype)[1:-1]
        alpha = alpha.view(-1, 1, 1, 1)
        linear_src = torch.pow(blend_src, gamma)
        linear_dst = torch.pow(blend_dst, gamma)
        blended = (1 - alpha) * linear_src + alpha * linear_dst
        blended_images = torch.pow(blended, 1.0 / gamma)

    elif mode == "perceptual_crossfade":
        import kornia # TODO: add to requirements.txt
        alpha = torch.linspace(0, 1, overlap + 2, device=blend_src.device, dtype=blend_src.dtype)[1:-1]

        src_nchw = blend_src.movedim(-1, 1)
        dst_nchw = blend_dst.movedim(-1, 1)
        lab_src = kornia.color.rgb_to_lab(src_nchw)
        lab_dst = kornia.color.rgb_to_lab(dst_nchw)

        # Blend in LAB space
        alpha = alpha.view(-1, 1, 1, 1)  # [N,1,1,1] for broadcasting
        blended_lab = (1 - alpha) * lab_src + alpha * lab_dst

        # Convert back to RGB and reshape
        blended_rgb = kornia.color.lab_to_rgb(blended_lab)
        blended_images = blended_rgb.movedim(1, -1)  # [N,C,H,W] -> [N,H,W,C]

    elif mode == "ease_in_out":
        t = torch.linspace(0, 1, overlap + 2, device=blend_src.device, dtype=blend_src.dtype)[1:-1]
        eased_t = 3 * t * t - 2 * t * t * t  # ease_in_out formula
        eased_t = eased_t.view(-1, 1, 1, 1)
        blended_images = (1 - eased_t) * blend_src + eased_t * blend_dst
    
    return blended_images