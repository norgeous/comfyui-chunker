import os
import sys
import torch

# ensure module imports work when running this script directly
sys.path.append(os.path.dirname(__file__))

from rgba import combine_images_masks

# Example inputs
# images = torch.rand(1, 256, 256, 3)  # BHWC
# masks = torch.rand(1, 256, 256)      # BHW

def make_dummy_frames(N=4, H=128, W=128):
    images = torch.zeros((N, H, W, 3), dtype=torch.float32)
    masks = torch.zeros((N, H, W), dtype=torch.float32)
    for i in range(N):
        # gradient image - brighter for visibility
        xv = torch.linspace(0, 1, W).unsqueeze(0).repeat(H, 1)
        yv = torch.linspace(0, 1, H).unsqueeze(1).repeat(1, W)
        r = xv.unsqueeze(-1)  # Red gradient left to right
        g = yv.unsqueeze(-1)  # Green gradient top to bottom
        b = 0.5 * torch.ones((H, W, 1), dtype=torch.float32)  # Constant blue
        images[i] = torch.cat([r, g, b], dim=-1)

        # circular mask
        cx, cy = W // 2, H // 2
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        masks[i] = (dist < (min(H, W) * (0.25 + 0.15 * i))).float()

    return images, masks

images, masks = make_dummy_frames(N=3, H=128, W=128)

combine_images_masks(images, masks, 'results/rgba')

