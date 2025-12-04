import os
import sys
import math
import numpy as np

# ensure module imports work when running this script directly
sys.path.append(os.path.dirname(__file__))

from prores_flac_saver import save_prores_with_alpha

def make_dummy_frames(N=4, H=128, W=128):
    import torch
    images = torch.zeros((N, H, W, 3), dtype=torch.float32)
    masks = torch.zeros((N, H, W), dtype=torch.float32)
    for i in range(N):
        # gradient image
        xv = torch.linspace(0, 1, W).unsqueeze(0).repeat(H, 1)
        yv = torch.linspace(0, 1, H).unsqueeze(1).repeat(1, W)
        r = (xv * (i + 1) / N).unsqueeze(-1)
        g = (yv * (i + 1) / N).unsqueeze(-1)
        b = (((xv + yv) * 0.5) * (i + 1) / N).unsqueeze(-1)
        images[i] = torch.cat([r, g, b], dim=-1)

        # circular mask
        cx, cy = W // 2, H // 2
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        masks[i] = (dist < (min(H, W) * (0.25 + 0.15 * i))).float()

    return images, masks

def make_dummy_audio(duration_s=1.0, sample_rate=22050):
    import torch
    t = torch.linspace(0, duration_s, int(duration_s * sample_rate), dtype=torch.float32)
    tone = 0.2 * torch.sin(2 * math.pi * 440.0 * t)
    stereo = torch.stack([tone, tone], dim=0)  # (2, S)
    waveform = stereo.unsqueeze(0)  # (1, 2, S)
    return waveform, sample_rate

def run_smoke():
    images, masks = make_dummy_frames(N=6, H=128, W=128)
    waveform, sr = make_dummy_audio(duration_s=1.0, sample_rate=22050)
    audio = {"waveform": waveform, "sample_rate": sr}

    out_path, meta = save_prores_with_alpha(images=images, masks=masks, audio=audio, fps=24, filename_prefix="smoke_test")
    print("Saved:", out_path)
    assert os.path.exists(out_path), f"Output file not found: {out_path}"
    print("Smoke test succeeded — file exists.")

if __name__ == '__main__':
    run_smoke()
