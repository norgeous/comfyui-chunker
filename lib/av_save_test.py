import os
import pytest
from lib.conftest import create_source_tensors
from lib.av_save import av_save, Profile


@pytest.fixture
def tensors_video_only():
    images, masks, audio = create_source_tensors(
        bg_color=(255, 0, 0), line_num=1, freq=440)
    return {"images": images, "masks": masks, "audio": audio}


@pytest.fixture
def tensors_audio_mono():
    images, masks, audio = create_source_tensors(
        bg_color=(255, 0, 0), line_num=1, freq=440, stereo=False)
    return {"images": images, "masks": masks, "audio": audio}


@pytest.fixture
def tensors_audio_stereo():
    images, masks, audio = create_source_tensors(
        bg_color=(255, 0, 0), line_num=1, freq=440, stereo=True)
    return {"images": images, "masks": masks, "audio": audio}


def test_video_only(output_dir, tensors_video_only):
    result, _ = av_save(
        images=tensors_video_only["images"],
        filename_prefix="save_video_only",
        fps=15)
    assert os.path.exists(result)


def test_audio_mono_only(output_dir, tensors_audio_mono):
    result, _ = av_save(
        audio=tensors_audio_mono["audio"],
        filename_prefix="save_audio_mono")
    assert os.path.exists(result)


def test_audio_stereo_only(output_dir, tensors_audio_stereo):
    result, _ = av_save(
        audio=tensors_audio_stereo["audio"],
        filename_prefix="save_audio_stereo")
    assert os.path.exists(result)


def test_video_and_audio_mono(output_dir, tensors_audio_mono):
    result, _ = av_save(
        images=tensors_audio_mono["images"],
        audio=tensors_audio_mono["audio"],
        filename_prefix="save_video_audio_mono",
        fps=15)
    assert os.path.exists(result)


def test_video_and_audio_stereo(output_dir, tensors_audio_stereo):
    result, _ = av_save(
        images=tensors_audio_stereo["images"],
        audio=tensors_audio_stereo["audio"],
        filename_prefix="save_video_audio_stereo",
        fps=15)
    assert os.path.exists(result)


def test_no_inputs_raises(output_dir):
    with pytest.raises(
            ValueError,
            match="At least one of images or audio "
                  "must be provided",
    ):
        av_save(filename_prefix="save_empty")


def test_web_profile_video_only(output_dir, tensors_video_only):
    result, _ = av_save(
        images=tensors_video_only["images"],
        filename_prefix="save_web_video",
        fps=15,
        profile=Profile.WEB)
    assert os.path.exists(result)
    assert result.endswith(".webm")


def test_web_profile_audio_mono(output_dir, tensors_audio_mono):
    result, _ = av_save(
        audio=tensors_audio_mono["audio"],
        filename_prefix="save_web_audio_mono",
        profile=Profile.WEB)
    assert os.path.exists(result)
    assert result.endswith(".webm")


def test_web_profile_video_and_audio_stereo(output_dir, tensors_audio_stereo):
    result, _ = av_save(
        images=tensors_audio_stereo["images"],
        audio=tensors_audio_stereo["audio"],
        filename_prefix="save_web_video_audio_stereo",
        fps=15,
        profile=Profile.WEB)
    assert os.path.exists(result)
    assert result.endswith(".webm")


def test_hq_profile_default_extension(output_dir, tensors_video_only):
    result, _ = av_save(
        images=tensors_video_only["images"],
        filename_prefix="save_hq_default",
        fps=15)
    assert result.endswith(".mp4")
