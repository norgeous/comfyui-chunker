import os
import pytest
from .av_load import av_load
from .av_save import av_save


def test_load_video_audio_stereo(source_videos):
    path = source_videos["source1"]
    images, audio, fps = av_load(path)

    assert images is not None
    assert images.shape == (30, 512, 512, 3), f"Expected 30 frames, got {images.shape[0]}"
    assert audio is not None
    assert audio["waveform"].shape[0] == 1
    assert audio["waveform"].shape[2] == 88200, f"Expected 88200 samples, got {audio['waveform'].shape[2]}"
    assert fps == 15, f"Expected 15 FPS, got {fps}"


def test_load_mono_audio_shape(source_videos):
    path = source_videos["source1"]
    _, audio, _ = av_load(path)
    assert audio is not None
    assert audio["waveform"].shape == (1, 1, 88200), f"Expected (1, 1, 88200), got {audio['waveform'].shape}"


def test_load_stereo_audio_shape(source_videos):
    path = source_videos["source1_stereo"]
    _, audio, _ = av_load(path)
    assert audio is not None
    assert audio["waveform"].shape == (1, 2, 88200), f"Expected (1, 2, 88200), got {audio['waveform'].shape}"


def test_save_load_mono_roundtrip(output_dir):
    audio = {
        "waveform": __import__('torch').randn(1, 1, 44100),
        "sample_rate": 44100,
    }
    path = os.path.join(output_dir, "roundtrip_mono")
    av_save(audio=audio, output_path=path)
    _, loaded, _ = av_load(f"{path}.mp4")
    assert loaded["waveform"].shape == (1, 1, 44100)


def test_save_load_stereo_roundtrip(output_dir):
    audio = {
        "waveform": __import__('torch').randn(1, 2, 44100),
        "sample_rate": 44100,
    }
    path = os.path.join(output_dir, "roundtrip_stereo")
    av_save(audio=audio, output_path=path)
    _, loaded, _ = av_load(f"{path}.mp4")
    assert loaded["waveform"].shape == (1, 2, 44100)


def test_load_nonexistent():
    with pytest.raises(Exception):
        images, audio, fps = av_load("nonexistent.mp4")
