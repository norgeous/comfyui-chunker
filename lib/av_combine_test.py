import pytest
import numpy as np
import os
from av_combine import av_combine, BlendMode
from conftest import get_frame_info, analyze_audio_frequency


def test_no_overlap(source_videos, output_dir):
    videos = [
        str(source_videos["source1"]),
        str(source_videos["source2"]),
        str(source_videos["source3"]),
    ]
    output_path = os.path.join(output_dir, "combine-3x30i-0o-none.mp4")

    av_combine(videos, output_path, 0, BlendMode.LINEAR, BlendMode.EQUAL_POWER)

    frames, audio_samples, _, _, _ = get_frame_info(output_path)

    assert len(frames) == 90, f"Expected 90 frames, got {len(frames)}"

    frames_array = np.array(frames)

    avg_bg_source1 = np.mean(frames_array[0:30, :, 300:], axis=(0, 1, 2))[:3]
    assert abs(avg_bg_source1[0] - 255) < 10 and avg_bg_source1[1] < 10 and avg_bg_source1[2] < 10, f"Frames 0-29 should be red"

    avg_bg_source2 = np.mean(frames_array[30:60, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_source2[0] < 10 and abs(avg_bg_source2[1] - 255) < 10 and avg_bg_source2[2] < 10, f"Frames 30-59 should be green"

    avg_bg_source3 = np.mean(frames_array[60:90, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_source3[0] < 10 and avg_bg_source3[1] < 10 and abs(avg_bg_source3[2] - 255) < 10, f"Frames 60-89 should be blue"

    audio_220 = analyze_audio_frequency(audio_samples[0:87])
    assert 200 < audio_220 < 240, f"Audio frames 0-86 should be ~220Hz, got {audio_220}"

    audio_440 = analyze_audio_frequency(audio_samples[87:174])
    assert 420 < audio_440 < 460, f"Audio frames 87-173 should be ~440Hz, got {audio_440}"

    audio_880 = analyze_audio_frequency(audio_samples[174:])
    assert 860 < audio_880 < 900, f"Audio frames 174+ should be ~880Hz, got {audio_880}"


def test_older_only(source_videos, output_dir):
    videos = [
        str(source_videos["source1"]),
        str(source_videos["source2"]),
        str(source_videos["source3"]),
    ]
    output_path = os.path.join(output_dir, "combine-3x30i-4o-older_only.mp4")

    av_combine(videos, output_path, 4, BlendMode.OLDER_ONLY, BlendMode.OLDER_ONLY)

    frames, audio_samples, _, _, _ = get_frame_info(output_path)

    assert len(frames) == 82, f"Expected 82 frames, got {len(frames)}"

    frames_array = np.array(frames)

    avg_bg_1_30 = np.mean(frames_array[0:30, :, 300:], axis=(0, 1, 2))[:3]
    assert abs(avg_bg_1_30[0] - 255) < 10 and avg_bg_1_30[1] < 10 and avg_bg_1_30[2] < 10, f"Frames 1-30 should be red"

    avg_bg_31_56 = np.mean(frames_array[30:56, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_31_56[0] < 10 and abs(avg_bg_31_56[1] - 255) < 10 and avg_bg_31_56[2] < 10, f"Frames 31-56 should be green"

    avg_bg_57_82 = np.mean(frames_array[56:82, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_57_82[0] < 10 and avg_bg_57_82[1] < 10 and abs(avg_bg_57_82[2] - 255) < 10, f"Frames 57-82 should be blue"

    audio_220 = analyze_audio_frequency(audio_samples[0:87])
    assert 200 < audio_220 < 240, f"Audio frames 0-86 should be ~220Hz, got {audio_220}"

    audio_440 = analyze_audio_frequency(audio_samples[87:174])
    assert 420 < audio_440 < 460, f"Audio frames 87-173 should be ~440Hz, got {audio_440}"

    audio_880 = analyze_audio_frequency(audio_samples[174:])
    assert 860 < audio_880 < 900, f"Audio frames 174+ should be ~880Hz, got {audio_880}"


def test_linear_blend(source_videos, output_dir):
    videos = [
        str(source_videos["source1"]),
        str(source_videos["source2"]),
        str(source_videos["source3"]),
    ]
    output_path = os.path.join(output_dir, "combine-3x30i-4o-linear.mp4")

    av_combine(videos, output_path, 4, BlendMode.LINEAR, BlendMode.LINEAR)

    frames, audio_samples, _, _, _ = get_frame_info(output_path)

    assert len(frames) == 82, f"Expected 82 frames, got {len(frames)}"

    frames_array = np.array(frames)

    avg_bg_1_26 = np.mean(frames_array[0:26, :, 300:], axis=(0, 1, 2))[:3]
    assert abs(avg_bg_1_26[0] - 255) < 10 and avg_bg_1_26[1] < 10 and avg_bg_1_26[2] < 10, f"Frames 1-26 should be red"

    frame_27 = frames[26]
    avg_27 = np.mean(frame_27[:, 300:], axis=(0, 1))[:3]
    expected_r = 255 * 0.8
    expected_g = 255 * 0.2
    assert abs(avg_27[0] - expected_r) < 20 and abs(avg_27[1] - expected_g) < 20, f"Frame 27 should be 80% red, 20% green"

    frame_28 = frames[27]
    avg_28 = np.mean(frame_28[:, 300:], axis=(0, 1))[:3]
    expected_r = 255 * 0.6
    expected_g = 255 * 0.4
    assert abs(avg_28[0] - expected_r) < 20 and abs(avg_28[1] - expected_g) < 20, f"Frame 28 should be 60% red, 40% green"

    frame_29 = frames[28]
    avg_29 = np.mean(frame_29[:, 300:], axis=(0, 1))[:3]
    expected_r = 255 * 0.4
    expected_g = 255 * 0.6
    assert abs(avg_29[0] - expected_r) < 20 and abs(avg_29[1] - expected_g) < 20, f"Frame 29 should be 40% red, 60% green"

    frame_30 = frames[29]
    avg_30 = np.mean(frame_30[:, 300:], axis=(0, 1))[:3]
    expected_r = 255 * 0.2
    expected_g = 255 * 0.8
    assert abs(avg_30[0] - expected_r) < 20 and abs(avg_30[1] - expected_g) < 20, f"Frame 30 should be 20% red, 80% green"

    avg_bg_31_52 = np.mean(frames_array[30:52, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_31_52[0] < 10 and abs(avg_bg_31_52[1] - 255) < 10 and avg_bg_31_52[2] < 10, f"Frames 31-52 should be green"

    avg_bg_57_82 = np.mean(frames_array[56:82, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_57_82[0] < 10 and avg_bg_57_82[1] < 10 and abs(avg_bg_57_82[2] - 255) < 10, f"Frames 57-82 should be blue"

    audio_220 = analyze_audio_frequency(audio_samples[0:80])
    assert 200 < audio_220 < 240, f"Audio start should be ~220Hz, got {audio_220}"

    audio_440 = analyze_audio_frequency(audio_samples[80:160])
    assert 420 < audio_440 < 460, f"Audio middle should be ~440Hz, got {audio_440}"

    audio_880 = analyze_audio_frequency(audio_samples[160:])
    assert 860 < audio_880 < 900, f"Audio end should be ~880Hz, got {audio_880}"


def test_ease_in_out(source_videos, output_dir):
    videos = [
        str(source_videos["source1"]),
        str(source_videos["source2"]),
        str(source_videos["source3"]),
    ]
    output_path = os.path.join(output_dir, "combine-3x30i-4o-ease_in_out.mp4")

    av_combine(videos, output_path, 4, BlendMode.EASE_IN_OUT, BlendMode.EASE_IN_OUT)

    frames, audio_samples, _, _, _ = get_frame_info(output_path)

    assert len(frames) == 82, f"Expected 82 frames, got {len(frames)}"

    frames_array = np.array(frames)

    avg_bg_1_26 = np.mean(frames_array[0:26, :, 300:], axis=(0, 1, 2))[:3]
    assert abs(avg_bg_1_26[0] - 255) < 10 and avg_bg_1_26[1] < 10 and avg_bg_1_26[2] < 10, f"Frames 1-26 should be red"

    expected_r_27 = 255 * 0.9045
    expected_g_27 = 255 * 0.0955
    frame_27 = frames[26]
    avg_27 = np.mean(frame_27[:, 300:], axis=(0, 1))[:3]
    assert abs(avg_27[0] - expected_r_27) < 20 and abs(avg_27[1] - expected_g_27) < 20, f"Frame 27 should be ~90.45% red, ~9.55% green"

    expected_r_28 = 255 * 0.6545
    expected_g_28 = 255 * 0.3455
    frame_28 = frames[27]
    avg_28 = np.mean(frame_28[:, 300:], axis=(0, 1))[:3]
    assert abs(avg_28[0] - expected_r_28) < 20 and abs(avg_28[1] - expected_g_28) < 20, f"Frame 28 should be ~65.45% red, ~34.55% green"

    expected_r_29 = 255 * 0.3455
    expected_g_29 = 255 * 0.6545
    frame_29 = frames[28]
    avg_29 = np.mean(frame_29[:, 300:], axis=(0, 1))[:3]
    assert abs(avg_29[0] - expected_r_29) < 20 and abs(avg_29[1] - expected_g_29) < 20, f"Frame 29 should be ~34.55% red, ~65.45% green"

    expected_r_30 = 255 * 0.0955
    expected_g_30 = 255 * 0.9045
    frame_30 = frames[29]
    avg_30 = np.mean(frame_30[:, 300:], axis=(0, 1))[:3]
    assert abs(avg_30[0] - expected_r_30) < 20 and abs(avg_30[1] - expected_g_30) < 20, f"Frame 30 should be ~9.55% red, ~90.45% green"

    avg_bg_31_52 = np.mean(frames_array[30:52, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_31_52[0] < 10 and abs(avg_bg_31_52[1] - 255) < 10 and avg_bg_31_52[2] < 10, f"Frames 31-52 should be green"

    avg_bg_57_82 = np.mean(frames_array[56:82, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_57_82[0] < 10 and avg_bg_57_82[1] < 10 and abs(avg_bg_57_82[2] - 255) < 10, f"Frames 57-82 should be blue"

    audio_220 = analyze_audio_frequency(audio_samples[0:80])
    assert 200 < audio_220 < 240, f"Audio start should be ~220Hz, got {audio_220}"

    audio_440 = analyze_audio_frequency(audio_samples[80:160])
    assert 420 < audio_440 < 460, f"Audio middle should be ~440Hz, got {audio_440}"

    audio_880 = analyze_audio_frequency(audio_samples[160:])
    assert 860 < audio_880 < 900, f"Audio end should be ~880Hz, got {audio_880}"


def test_newer_only(source_videos, output_dir):
    videos = [
        str(source_videos["source1"]),
        str(source_videos["source2"]),
        str(source_videos["source3"]),
    ]
    output_path = os.path.join(output_dir, "combine-3x30i-4o-newer_only.mp4")

    av_combine(videos, output_path, 4, BlendMode.NEWER_ONLY, BlendMode.NEWER_ONLY)

    frames, audio_samples, _, _, _ = get_frame_info(output_path)

    assert len(frames) == 82, f"Expected 82 frames, got {len(frames)}"

    frames_array = np.array(frames)

    avg_bg_0_25 = np.mean(frames_array[0:26, :, 300:], axis=(0, 1, 2))[:3]
    assert abs(avg_bg_0_25[0] - 255) < 10 and avg_bg_0_25[1] < 10 and avg_bg_0_25[2] < 10, f"Frames 0-25 should be red"

    avg_bg_30_51 = np.mean(frames_array[30:52, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_30_51[0] < 10 and abs(avg_bg_30_51[1] - 255) < 10 and avg_bg_30_51[2] < 10, f"Frames 30-51 should be green"

    avg_bg_60_81 = np.mean(frames_array[60:82, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_60_81[0] < 10 and avg_bg_60_81[1] < 10 and abs(avg_bg_60_81[2] - 255) < 10, f"Frames 60-81 should be blue"

    audio_220 = analyze_audio_frequency(audio_samples[0:87])
    assert 200 < audio_220 < 240, f"Audio frames 0-86 should be ~220Hz, got {audio_220}"

    audio_440 = analyze_audio_frequency(audio_samples[87:174])
    assert 420 < audio_440 < 460, f"Audio frames 87-173 should be ~440Hz, got {audio_440}"

    audio_880 = analyze_audio_frequency(audio_samples[174:])
    assert 860 < audio_880 < 900, f"Audio frames 174+ should be ~880Hz, got {audio_880}"


def test_equal_power(source_videos, output_dir):
    videos = [
        str(source_videos["source1"]),
        str(source_videos["source2"]),
        str(source_videos["source3"]),
    ]
    output_path = os.path.join(output_dir, "combine-3x30i-4o-equal_power.mp4")

    av_combine(videos, output_path, 4, BlendMode.EQUAL_POWER, BlendMode.EQUAL_POWER)

    frames, audio_samples, _, _, _ = get_frame_info(output_path)

    assert len(frames) == 82, f"Expected 82 frames, got {len(frames)}"

    frames_array = np.array(frames)

    avg_bg_1_26 = np.mean(frames_array[0:26, :, 300:], axis=(0, 1, 2))[:3]
    assert abs(avg_bg_1_26[0] - 255) < 10 and avg_bg_1_26[1] < 10 and avg_bg_1_26[2] < 10, f"Frames 1-26 should be red"

    frame_27 = frames[26]
    avg_27 = np.mean(frame_27[:, 300:], axis=(0, 1))[:3]
    expected_r = 255 * 0.691
    expected_g = 255 * 0.309
    assert abs(avg_27[0] - expected_r) < 20 and abs(avg_27[1] - expected_g) < 20, f"Frame 27 should be ~69.1% red, ~30.9% green"

    frame_28 = frames[27]
    avg_28 = np.mean(frame_28[:, 300:], axis=(0, 1))[:3]
    expected_r = 255 * 0.412
    expected_g = 255 * 0.588
    assert abs(avg_28[0] - expected_r) < 20 and abs(avg_28[1] - expected_g) < 20, f"Frame 28 should be ~41.2% red, ~58.8% green"

    frame_29 = frames[28]
    avg_29 = np.mean(frame_29[:, 300:], axis=(0, 1))[:3]
    expected_r = 255 * 0.191
    expected_g = 255 * 0.809
    assert abs(avg_29[0] - expected_r) < 20 and abs(avg_29[1] - expected_g) < 20, f"Frame 29 should be ~19.1% red, ~80.9% green"

    frame_30 = frames[29]
    avg_30 = np.mean(frame_30[:, 300:], axis=(0, 1))[:3]
    expected_r = 255 * 0.049
    expected_g = 255 * 0.951
    assert abs(avg_30[0] - expected_r) < 20 and abs(avg_30[1] - expected_g) < 20, f"Frame 30 should be ~4.9% red, ~95.1% green"

    avg_bg_31_52 = np.mean(frames_array[30:52, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_31_52[0] < 10 and abs(avg_bg_31_52[1] - 255) < 10 and avg_bg_31_52[2] < 10, f"Frames 31-52 should be green"

    avg_bg_57_82 = np.mean(frames_array[56:82, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_57_82[0] < 10 and avg_bg_57_82[1] < 10 and abs(avg_bg_57_82[2] - 255) < 10, f"Frames 57-82 should be blue"

    audio_220 = analyze_audio_frequency(audio_samples[0:80])
    assert 200 < audio_220 < 240, f"Audio start should be ~220Hz, got {audio_220}"

    audio_440 = analyze_audio_frequency(audio_samples[80:160])
    assert 420 < audio_440 < 460, f"Audio middle should be ~440Hz, got {audio_440}"

    audio_880 = analyze_audio_frequency(audio_samples[160:])
    assert 860 < audio_880 < 900, f"Audio end should be ~880Hz, got {audio_880}"


def test_stereo_linear_blend(source_videos, output_dir):
    videos = [
        str(source_videos["source1_stereo"]),
        str(source_videos["source2_stereo"]),
        str(source_videos["source3_stereo"]),
    ]
    output_path = os.path.join(output_dir, "combine-3x30i-4o-stereo_linear.mp4")

    av_combine(videos, output_path, 4, BlendMode.LINEAR, BlendMode.LINEAR)

    frames, _, audio_left, audio_right, _ = get_frame_info(output_path)

    assert len(frames) == 82, f"Expected 82 frames, got {len(frames)}"

    frames_array = np.array(frames)

    avg_bg_1_26 = np.mean(frames_array[0:26, :, 300:], axis=(0, 1, 2))[:3]
    assert abs(avg_bg_1_26[0] - 255) < 10 and avg_bg_1_26[1] < 10 and avg_bg_1_26[2] < 10, f"Frames 1-26 should be red"

    avg_bg_31_52 = np.mean(frames_array[30:52, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_31_52[0] < 10 and abs(avg_bg_31_52[1] - 255) < 10 and avg_bg_31_52[2] < 10, f"Frames 31-52 should be green"

    avg_bg_57_82 = np.mean(frames_array[56:82, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_57_82[0] < 10 and avg_bg_57_82[1] < 10 and abs(avg_bg_57_82[2] - 255) < 10, f"Frames 57-82 should be blue"

    assert audio_left is not None and audio_right is not None, "Expected stereo audio output"

    audio_220_left = analyze_audio_frequency(audio_left[0:2940 * 30])
    audio_440_left = analyze_audio_frequency(audio_left[2940 * 30:2940 * 56])
    audio_880_left = analyze_audio_frequency(audio_left[2940 * 56:])
    assert 200 < audio_220_left < 240, f"Left channel start should be ~220Hz, got {audio_220_left}"
    assert 420 < audio_440_left < 460, f"Left channel middle should be ~440Hz, got {audio_440_left}"
    assert 860 < audio_880_left < 900, f"Left channel end should be ~880Hz, got {audio_880_left}"

    audio_220_right = analyze_audio_frequency(audio_right[0:2940 * 30])
    audio_440_right = analyze_audio_frequency(audio_right[2940 * 30:2940 * 56])
    audio_880_right = analyze_audio_frequency(audio_right[2940 * 56:])
    assert 200 < audio_220_right < 240, f"Right channel start should be ~220Hz, got {audio_220_right}"
    assert 420 < audio_440_right < 460, f"Right channel middle should be ~440Hz, got {audio_440_right}"
    assert 860 < audio_880_right < 900, f"Right channel end should be ~880Hz, got {audio_880_right}"
