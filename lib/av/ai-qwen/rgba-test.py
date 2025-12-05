import os
import sys
import torch
import math

# ensure module imports work when running this script directly
sys.path.append(os.path.dirname(__file__))

from rgba import combine_images_masks

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

def make_dummy_audio(duration_s=1.0, sample_rate=22050):
    t = torch.linspace(0, duration_s, int(duration_s * sample_rate), dtype=torch.float32)
    tone = 0.2 * torch.sin(2 * math.pi * 440.0 * t)
    stereo = torch.stack([tone, tone], dim=0)  # (2, S)
    waveform = stereo.unsqueeze(0)  # (1, 2, S)
    # Ensure values are in -1..1 range for audio (will be normalized by encoder)
    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }

images, masks = make_dummy_frames(N=10, H=128, W=128)
audio = make_dummy_audio(duration_s=1.0, sample_rate=22050)
fps = 10

combine_images_masks('results/rgba', images, masks, audio, fps)

