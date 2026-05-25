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
    mask = image[:, :, :, 0] # keep first 3 dimensions, choose index 0 (red) for new channel
    return mask


def resize_image(image: Optional[torch.Tensor],
                 width: int,
                 height: int) -> Optional[torch.Tensor]:
    if image is None:
        return None
    if image.shape[1] == height and image.shape[2] == width:
        return image
    resized_image = (
        common_upscale(
            image.movedim(-1, 1), width, height, "lanczos", "center",
        )
        .movedim(1, -1)
    )
    return resized_image


def resize_mask(mask: Optional[torch.Tensor],
                width: int,
                height: int) -> Optional[torch.Tensor]:
    imask = mask_to_image(mask)
    resized_imask = resize_image(imask, width, height)
    return image_to_mask(resized_imask)


def simple_blend(image1: torch.Tensor, image2: torch.Tensor,
                 blend_factor: float = 0.5) -> torch.Tensor:
    blended_image = image1 * (1 - blend_factor) + image2 * blend_factor
    blended_image = torch.clamp(blended_image, 0, 1)
    return blended_image
