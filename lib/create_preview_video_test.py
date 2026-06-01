import os
from lib.conftest import create_source_tensors
from lib.create_preview_video import create_preview_video
from lib.av_save import av_save, Profile


def test_create_preview_video_with_images_masks_audio(output_dir):
    images, masks, audio = create_source_tensors()
    d = {"index": 0, "fps": 15.0}
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
