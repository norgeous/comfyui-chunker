import pytest
import torch
from lib.av_load import av_load
from lib.av_save import av_save, Profile
from lib.conftest import create_source_tensors, analyze_audio_frequency


def test_load_video_audio_stereo(source_videos):
    path = source_videos["source1"]
    images, masks, audio, fps = av_load(path)

    assert images is not None
    assert images.shape == (
        30, 512, 512, 3), f"Expected 30 frames, got {
        images.shape[0]}"
    assert masks is None
    assert audio is not None
    assert audio["waveform"].shape[0] == 1
    assert audio["waveform"].shape[2] == 88200, f"Expected 88200 samples, got {
        audio['waveform'].shape[2]}"
    assert fps == 15, f"Expected 15 FPS, got {fps}"


def test_load_mono_audio_shape(source_videos):
    path = source_videos["source1"]
    _, _, audio, _ = av_load(path)
    assert audio is not None
    assert audio["waveform"].shape == (
        1, 1, 88200), f"Expected (1, 1, 88200), got {
        audio['waveform'].shape}"


def test_load_stereo_audio_shape(source_videos):
    path = source_videos["source1_stereo"]
    _, _, audio, _ = av_load(path)
    assert audio is not None
    assert audio["waveform"].shape == (
        1, 2, 88200), f"Expected (1, 2, 88200), got {
        audio['waveform'].shape}"


@pytest.mark.parametrize("profile", [Profile.HQ, Profile.WEBRGB, Profile.WEBRGBA])
def test_save_load_mono_roundtrip(profile):
    freq = 440
    images, masks, audio = create_source_tensors(bg_color=(
        255, 0, 0), line_num=1, freq=freq, stereo=False, audio_duration=1.0)
    saved_path, _ = av_save(audio=audio, filename_prefix="roundtrip_mono", profile=profile)
    _, _, loaded, _ = av_load(saved_path)
    assert loaded["waveform"].shape[0] == 1
    assert loaded["waveform"].shape[1] == 1
    if profile == Profile.HQ:
        assert loaded["waveform"].shape[2] == audio["waveform"].shape[2]
    else:
        assert abs(
            loaded["waveform"].shape[2] -
            audio["waveform"].shape[2]) < 1000
    samples = loaded["waveform"].squeeze().numpy()
    detected_freq = analyze_audio_frequency(samples, loaded["sample_rate"])
    assert abs(detected_freq -
               freq) < 10, f"Expected {freq}Hz, got {detected_freq}Hz"


@pytest.mark.parametrize("profile", [Profile.HQ, Profile.WEBRGB, Profile.WEBRGBA])
def test_save_load_stereo_roundtrip(profile):
    freq = 440
    images, masks, audio = create_source_tensors(bg_color=(
        255, 0, 0), line_num=1, freq=freq, stereo=True, audio_duration=1.0)
    saved_path, _ = av_save(audio=audio, filename_prefix="roundtrip_stereo", profile=profile)
    _, _, loaded, _ = av_load(saved_path)
    assert loaded["waveform"].shape[0] == 1
    assert loaded["waveform"].shape[1] == 2
    if profile == Profile.HQ:
        assert loaded["waveform"].shape[2] == audio["waveform"].shape[2]
    else:
        assert abs(
            loaded["waveform"].shape[2] -
            audio["waveform"].shape[2]) < 1000
    for channel in range(loaded["waveform"].shape[1]):
        samples = loaded["waveform"][0, channel].numpy()
        detected_freq = analyze_audio_frequency(samples, loaded["sample_rate"])
        assert abs(detected_freq - freq) < 10, (
            f"Channel {channel}: Expected {freq}Hz, got {detected_freq}Hz"
        )


def test_load_nonexistent():
    with pytest.raises(Exception):
        av_load("nonexistent.mp4")


def test_av_load_window_matches_av_load(source_videos):
    path = source_videos["source-long"]
    fi, fm, fa, fps = av_load(path)
    spf = 2940  # floor(44100 / 15)

    for start, end in [(0, 10), (10, 25), (45, 60), (50, 90)]:
        wi, wm, wa, wf = av_load(path, start=start, end=end)
        exp_end = min(end, len(fi))
        assert wf == fps
        assert wm is None
        assert torch.equal(wi, fi[start:exp_end]), f"frames {start}:{end}"
        assert wa is not None
        assert torch.equal(wa["waveform"],
                           fa["waveform"][:, :, start * spf:exp_end * spf])


def test_av_load_open_ended(source_videos):
    path = source_videos["source-long"]
    fi, _, fa, _ = av_load(path)
    spf = 2940

    wi, wm, wa, _ = av_load(path, start=30)
    assert wm is None
    assert torch.equal(wi, fi[30:])
    assert torch.equal(wa["waveform"], fa["waveform"][:, :, 30 * spf:])


def test_av_load_past_eof(source_videos):
    path = source_videos["source-long"]

    wi, wm, wa, fps = av_load(path, start=1000, end=1100)
    assert fps == 15
    assert wi is None
    assert wm is None
    assert wa is not None
    assert wa["waveform"].shape[2] == 0


def test_av_load_window_mask_stream():
    images, masks, _ = create_source_tensors(audio_duration=0.0)
    saved_path, _ = av_save(
        images=images,
        masks=masks,
        filename_prefix="window-masks",
        profile=Profile.WEBRGBA,
    )
    fi, fm, _, _ = av_load(saved_path)

    wi, wm, _, _ = av_load(saved_path, start=5, end=20)
    assert torch.equal(wi, fi[5:20])
    assert torch.equal(wm, fm[5:20])


def test_av_load_negative_start_tail(source_videos):
    path = source_videos["source-long"]
    fi, fm, fa, fps = av_load(path)
    spf = 2940

    wi, wm, wa, wf = av_load(path, start=-7)
    assert wf == fps
    assert wm is None
    assert torch.equal(wi, fi[-7:])
    assert wa is not None
    assert torch.equal(wa["waveform"], fa["waveform"][:, :, -7 * spf:])


def test_av_load_negative_start_clamp(source_videos):
    path = source_videos["source-long"]

    wi, wm, wa, wf = av_load(path, start=-100)
    assert wf == fps
    assert wm is None
    assert wi is not None
    assert torch.equal(wi, fi)
    assert wa is not None
    assert torch.equal(wa["waveform"], fa["waveform"])


def test_av_load_negative_start_end(source_videos):
    path = source_videos["source-long"]
    fi, fm, fa, fps = av_load(path)
    spf = 2940

    wi, wm, wa, wf = av_load(path, start=-10, end=-4)
    assert wf == fps
    assert wm is None
    assert torch.equal(wi, fi[-10:-4])
    assert wa is not None
    assert torch.equal(wa["waveform"], fa["waveform"][:, :, -10 * spf:-4 * spf])


def test_av_load_negative_end(source_videos):
    path = source_videos["source-long"]
    fi, fm, fa, fps = av_load(path)
    spf = 2940

    wi, wm, wa, wf = av_load(path, start=40, end=-5)
    assert wf == fps
    assert wm is None
    assert torch.equal(wi, fi[40:-5])
    assert wa is not None
    assert torch.equal(wa["waveform"], fa["waveform"][:, :, 40 * spf:-5 * spf])


def test_av_load_negative_start_mask_stream():
    images, masks, _ = create_source_tensors(audio_duration=0.0)
    saved_path, _ = av_save(
        images=images,
        masks=masks,
        filename_prefix="negative-masks",
        profile=Profile.WEBRGBA,
    )
    fi, fm, _, _ = av_load(saved_path)

    wi, wm, _, _ = av_load(saved_path, start=-6)
    assert torch.equal(wi, fi[-6:])
    assert torch.equal(wm, fm[-6:])
