import os
import pytest
import torch
from .av_load import av_load
from .av_save import av_save, Profile
from .conftest import create_source_tensors


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


@pytest.mark.parametrize("profile", [Profile.HQ, Profile.WEB])
def test_save_load_mono_roundtrip(output_dir, profile):
    tensors = create_source_tensors(bg_color=(255, 0, 0), line_num=1, freq=440, stereo=False, audio_duration=1.0)
    audio = tensors["audio"]
    path = os.path.join(output_dir, "roundtrip_mono")
    saved_path = av_save(audio=audio, output_path=path, profile=profile)
    _, loaded, _ = av_load(saved_path)
    assert loaded["waveform"].shape[0] == 1
    assert loaded["waveform"].shape[1] == 1
    if profile == Profile.HQ:
        assert loaded["waveform"].shape[2] == audio["waveform"].shape[2]
    else:
        assert abs(loaded["waveform"].shape[2] - audio["waveform"].shape[2]) < 1000


@pytest.mark.parametrize("profile", [Profile.HQ, Profile.WEB])
def test_save_load_stereo_roundtrip(output_dir, profile):
    tensors = create_source_tensors(bg_color=(255, 0, 0), line_num=1, freq=440, stereo=True, audio_duration=1.0)
    audio = tensors["audio"]
    path = os.path.join(output_dir, "roundtrip_stereo")
    saved_path = av_save(audio=audio, output_path=path, profile=profile)
    _, loaded, _ = av_load(saved_path)
    assert loaded["waveform"].shape[0] == 1
    assert loaded["waveform"].shape[1] == 2
    if profile == Profile.HQ:
        assert loaded["waveform"].shape[2] == audio["waveform"].shape[2]
    else:
        assert abs(loaded["waveform"].shape[2] - audio["waveform"].shape[2]) < 1000


def test_load_nonexistent():
    with pytest.raises(Exception):
        images, audio, fps = av_load("nonexistent.mp4")
