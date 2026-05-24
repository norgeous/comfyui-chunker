import os
import torch
from conftest import create_source_tensors
from create_preview_video import create_preview_video
from av_save import av_save, Profile


def test_create_preview_video_with_images_masks_audio(output_dir):
    images, masks, audio = create_source_tensors(
        bg_color=(128, 128, 128),
        line_num=1,
        freq=440,
        video_frames=10,
        video_width=512,
        video_height=512,
    )

    d = {"index": 2, "fps": 24.0}
    c = {"chunk_length": 10, "chunk_overlap": 2, "chunk_count": 5, "total_length": 50}
    overlap_blend_mode = "linear"

    result = create_preview_video(images, masks, audio, d, c, overlap_blend_mode)

    assert result is not None
    assert result.shape == (10, 512, 512, 3)
    assert result.dtype == torch.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0

    output_path = os.path.join(output_dir, "preview_test.webm")
    av_save(images=result, audio=audio, output_path=output_path, fps=24.0, profile=Profile.WEB)
    assert os.path.exists(output_path)
