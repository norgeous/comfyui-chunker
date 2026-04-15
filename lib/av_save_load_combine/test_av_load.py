import pytest
from av_load import av_load


def test_load_video_audio_stereo(source_videos):
    path = source_videos["source1"]
    images, audio = av_load(path)

    assert images is not None
    assert images.shape == (30, 512, 512, 3), f"Expected 30 frames, got {images.shape[0]}"
    assert audio is not None
    assert audio["waveform"].shape[0] == 1
    assert audio["waveform"].shape[1] == 88200, f"Expected 88200 samples, got {audio['waveform'].shape[1]}"


def test_load_nonexistent():
    with pytest.raises(Exception):
        av_load("nonexistent.mp4")
