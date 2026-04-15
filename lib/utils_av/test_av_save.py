import os
import pytest
from .conftest import create_source_tensors
from .av_save import av_save


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