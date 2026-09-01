import os
import numpy as np
import torch
from lib.conftest import create_source_tensors
from lib.create_preview_video import create_preview_video, overlay_debug_text
from lib.av_save import av_save, Profile


def test_create_preview_video_with_images_masks_audio():
    images, masks, audio = create_source_tensors()
    d = {"index": 0, "fps": 15.0, "seed_info": ""}
    c = {
        "chunk_length": 30,
        "chunk_overlap": 10,
        "chunk_count": 2,
        "total_length": 50}
    overlap_blend_mode = "linear"
    preview, preview_masks, _, _ = create_preview_video(
        images, masks, audio, d, c, overlap_blend_mode)
    saved_path, _ = av_save(
        images=preview,
        masks=preview_masks,
        audio=audio,
        filename_prefix="preview_test",
        fps=15.0,
        profile=Profile.WEBRGBA)
    assert os.path.exists(saved_path)


def test_preview_overlay_bottom_right_padding():
    h, w = 512, 512
    rgba = torch.cat(
        [torch.zeros(1, h, w, 3), torch.ones(1, h, w, 1)], dim=-1)
    out = overlay_debug_text(
        rgba, 0, 0, 2, 30, 10, 50, 15.0, "linear", "44100Hz stereo",
        "seed: 12345")
    rgb = out[0, :, :, :3].numpy()
    mask = rgb.max(axis=2) > 0.05
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    bottom_gap = h - 1 - rows.max()
    right_gap = w - 1 - cols.max()
    expected = int((h / 512) * 10)
    assert bottom_gap >= expected - 1
    assert right_gap >= expected - 1
    assert abs(right_gap - bottom_gap) <= 1
