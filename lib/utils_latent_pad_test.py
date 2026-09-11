import torch
from comfy.nested_tensor import NestedTensor
from lib.utils_latent_pad import pad_latent_to_length


def _empty_video(tokens, width, height, batch, device):
    return torch.zeros([batch, 24, tokens, height // 16, width // 16], device=device)


def _empty_audio(tokens, batch, device):
    return torch.zeros([batch, 32, 2, tokens], device=device)

DEVICE = torch.device("cpu")


def test_video_only_padded():
    vid = torch.zeros(1, 24, 5, 4, 4)
    out, vp, ap = pad_latent_to_length(vid, 10, 0, _empty_video, _empty_audio, DEVICE, 64, 64)
    assert vp == 5
    assert ap == 0
    assert isinstance(out, torch.Tensor) and not hasattr(out, "tensors")
    assert tuple(out.shape) == (1, 24, 10, 4, 4)
    assert bool((out[:, :, :5] == 0).all())
    assert bool((out[:, :, 5:] == 0).all())


def test_video_exact_size():
    vid = torch.zeros(1, 24, 10, 4, 4)
    out, vp, ap = pad_latent_to_length(vid, 10, 0, _empty_video, _empty_audio, DEVICE, 64, 64)
    assert vp == 0 and ap == 0
    assert out is vid


def test_video_longer_never_truncated():
    vid = torch.ones(1, 24, 15, 4, 4)
    out, vp, ap = pad_latent_to_length(vid, 10, 0, _empty_video, _empty_audio, DEVICE, 64, 64)
    assert vp == 0 and ap == 0
    assert tuple(out.shape) == (1, 24, 15, 4, 4)


def test_nested_video_short_audio_short():
    vid = torch.zeros(1, 24, 5, 4, 4)
    aud = torch.zeros(1, 32, 2, 20)
    nt = NestedTensor((vid, aud))
    out, vp, ap = pad_latent_to_length(nt, 10, 40, _empty_video, _empty_audio, DEVICE, 64, 64)
    assert vp == 5 and ap == 20
    assert hasattr(out, "tensors")
    v, a = out.tensors
    assert tuple(v.shape) == (1, 24, 10, 4, 4)
    assert tuple(a.shape) == (1, 32, 2, 40)


def test_nested_video_only_audio_missing():
    vid = torch.zeros(1, 24, 10, 4, 4)
    nt = NestedTensor((vid,))
    out, vp, ap = pad_latent_to_length(nt, 10, 40, _empty_video, _empty_audio, DEVICE, 64, 64)
    assert vp == 0 and ap == 40
    assert hasattr(out, "tensors")
    v, a = out.tensors
    assert tuple(a.shape) == (1, 32, 2, 40)


def test_none_latent_creates_both():
    out, vp, ap = pad_latent_to_length(None, 10, 40, _empty_video, _empty_audio, DEVICE, 64, 64)
    assert vp == 10 and ap == 40
    assert hasattr(out, "tensors")
    v, a = out.tensors
    assert tuple(v.shape) == (1, 24, 10, 4, 4)
    assert tuple(a.shape) == (1, 32, 2, 40)


def test_none_latent_video_only_mode():
    out, vp, ap = pad_latent_to_length(None, 10, 0, _empty_video, _empty_audio, DEVICE, 64, 64)
    assert vp == 10 and ap == 0
    assert isinstance(out, torch.Tensor) and not hasattr(out, "tensors")
    assert tuple(out.shape) == (1, 24, 10, 4, 4)


def test_no_audio_mode_drops_nested_audio_stream():
    vid = torch.zeros(1, 24, 10, 4, 4)
    aud = torch.zeros(1, 32, 2, 40)
    nt = NestedTensor((vid, aud))
    out, vp, ap = pad_latent_to_length(nt, 10, 0, _empty_video, _empty_audio, DEVICE, 64, 64)
    assert ap == 0
    assert not hasattr(out, "tensors")
    assert tuple(out.shape) == (1, 24, 10, 4, 4)


def test_none_audio_fn_returns_none():
    out, vp, ap = pad_latent_to_length(None, 10, 0, _empty_video, lambda t, b, d: None, DEVICE, 64, 64)
    assert not hasattr(out, "tensors")
    assert tuple(out.shape) == (1, 24, 10, 4, 4)


def test_batch_preserved():
    vid = torch.zeros(2, 24, 3, 8, 8)
    out, vp, ap = pad_latent_to_length(vid, 10, 0, _empty_video, _empty_audio, DEVICE, 128, 128)
    assert vp == 7
    assert tuple(out.shape) == (2, 24, 10, 8, 8)
