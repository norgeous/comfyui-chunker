import os
import pytest
from conftest import create_source_tensors
from utils_av.av_save import av_save, Profile


@pytest.fixture
def tensors_video_only():
    return create_source_tensors(bg_color=(255, 0, 0), line_num=1, freq=440)


@pytest.fixture
def tensors_audio_mono():
    return create_source_tensors(bg_color=(255, 0, 0), line_num=1, freq=440, stereo=False)


@pytest.fixture
def tensors_audio_stereo():
    return create_source_tensors(bg_color=(255, 0, 0), line_num=1, freq=440, stereo=True)


def test_video_only(output_dir, tensors_video_only):
    output_path = os.path.join(output_dir, "save_video_only")
    result = av_save(images=tensors_video_only["images"], output_path=output_path, fps=15)
    assert os.path.exists(result)


def test_audio_mono_only(output_dir, tensors_audio_mono):
    output_path = os.path.join(output_dir, "save_audio_mono")
    result = av_save(audio=tensors_audio_mono["audio"], output_path=output_path)
    assert os.path.exists(result)


def test_audio_stereo_only(output_dir, tensors_audio_stereo):
    output_path = os.path.join(output_dir, "save_audio_stereo")
    result = av_save(audio=tensors_audio_stereo["audio"], output_path=output_path)
    assert os.path.exists(result)


def test_video_and_audio_mono(output_dir, tensors_audio_mono):
    output_path = os.path.join(output_dir, "save_video_audio_mono")
    result = av_save(images=tensors_audio_mono["images"], audio=tensors_audio_mono["audio"], output_path=output_path, fps=15)
    assert os.path.exists(result)


def test_video_and_audio_stereo(output_dir, tensors_audio_stereo):
    output_path = os.path.join(output_dir, "save_video_audio_stereo")
    result = av_save(images=tensors_audio_stereo["images"], audio=tensors_audio_stereo["audio"], output_path=output_path, fps=15)
    assert os.path.exists(result)


def test_no_inputs_raises(output_dir):
    with pytest.raises(ValueError, match="At least one of images or audio must be provided"):
        av_save(output_path=os.path.join(output_dir, "save_empty"))


def test_web_profile_video_only(output_dir, tensors_video_only):
    output_path = os.path.join(output_dir, "save_web_video")
    result = av_save(images=tensors_video_only["images"], output_path=output_path, fps=15, profile=Profile.WEB)
    assert os.path.exists(result)
    assert result.endswith(".webm")


def test_web_profile_audio_mono(output_dir, tensors_audio_mono):
    output_path = os.path.join(output_dir, "save_web_audio_mono")
    result = av_save(audio=tensors_audio_mono["audio"], output_path=output_path, profile=Profile.WEB)
    assert os.path.exists(result)
    assert result.endswith(".webm")


def test_web_profile_video_and_audio_stereo(output_dir, tensors_audio_stereo):
    output_path = os.path.join(output_dir, "save_web_video_audio_stereo")
    result = av_save(images=tensors_audio_stereo["images"], audio=tensors_audio_stereo["audio"], output_path=output_path, fps=15, profile=Profile.WEB)
    assert os.path.exists(result)
    assert result.endswith(".webm")


def test_hq_profile_default_extension(output_dir, tensors_video_only):
    output_path = os.path.join(output_dir, "save_hq_default")
    result = av_save(images=tensors_video_only["images"], output_path=output_path, fps=15)
    assert result.endswith(".mp4")
