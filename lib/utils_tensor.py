from typing import Optional
import torch
from comfy.utils import common_upscale


def mask_to_image(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    image = (
        mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        .movedim(1, -1)
        .expand(-1, -1, -1, 3)
    )
    return image


def image_to_mask(image: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if image is None:
        return None
    mask = image[:, :, :, 0]
    return mask


def resize_image(image: Optional[torch.Tensor], width: int, height: int, pad: bool = False) -> Optional[torch.Tensor]:
    if image is None:
        return None
    if image.shape[1] == height and image.shape[2] == width:
        return image

    samples = image.movedim(-1, 1)

    if pad:
        _, _, h, w = samples.shape
        target_aspect = width / height
        current_aspect = w / h

        if current_aspect > target_aspect:
            new_h = round(w / target_aspect)
            pad_top = (new_h - h) // 2
            pad_bottom = new_h - h - pad_top
            samples = torch.nn.functional.pad(samples, (0, 0, pad_top, pad_bottom), value=0.0)
        elif current_aspect < target_aspect:
            new_w = round(h * target_aspect)
            pad_left = (new_w - w) // 2
            pad_right = new_w - w - pad_left
            samples = torch.nn.functional.pad(samples, (pad_left, pad_right, 0, 0), value=0.0)

        resized = common_upscale(samples, width, height, "lanczos", "disabled")
    else:
        resized = common_upscale(samples, width, height, "lanczos", "center")

    return resized.movedim(1, -1)


def resize_mask(mask: Optional[torch.Tensor], width: int, height: int, pad: bool = False) -> Optional[torch.Tensor]:
    imask = mask_to_image(mask)
    resized_imask = resize_image(imask, width, height, pad)
    return image_to_mask(resized_imask)
