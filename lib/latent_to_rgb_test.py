import torch
import pytest

from lib.latent_to_rgb import latent_to_images

pytest.importorskip("comfy.nested_tensor")


def test_h3_nested_latent():
    from comfy.nested_tensor import NestedTensor
    video = torch.randn(1, 24, 32, 8, 22)
    audio = torch.randn(1, 32, 2, 85)
    latent = {"samples": NestedTensor([video, audio]), "type": "h3"}
    images = latent_to_images(latent, "minimax-h3")
    assert images is not None
    assert tuple(images.shape) == (32, 8 * 16, 22 * 16, 3)
    assert bool((images >= 0).all() and (images <= 1).all())
    assert not torch.isnan(images).any()
    assert not torch.isinf(images).any()


def test_wan2_16ch():
    video = torch.randn(1, 16, 21, 6, 8)
    images = latent_to_images({"samples": video}, "wan2")
    assert tuple(images.shape) == (21, 6 * 8, 8 * 8, 3)


def test_wan2_48ch():
    video = torch.randn(1, 48, 21, 4, 4)
    images = latent_to_images({"samples": video}, "wan2")
    assert tuple(images.shape) == (21, 4 * 16, 4 * 16, 3)


def test_ltx2_128ch():
    video = torch.randn(1, 128, 11, 2, 3)
    images = latent_to_images({"samples": video}, "ltx2")
    assert tuple(images.shape) == (11, 2 * 32, 3 * 32, 3)


def test_default_image_latent_4d():
    latent = torch.randn(1, 4, 8, 8)
    images = latent_to_images({"samples": latent}, "default")
    assert tuple(images.shape) == (1, 64, 64, 3)


def test_target_resolution():
    video = torch.randn(1, 16, 21, 6, 8)
    images = latent_to_images({"samples": video}, "wan2", target_w=512, target_h=320)
    assert tuple(images.shape) == (21, 320, 512, 3)


def test_none_for_non_video_latent():
    from comfy.nested_tensor import NestedTensor
    latent = {"samples": NestedTensor([torch.randn(1, 32, 2, 85)]), "type": "h3"}
    assert latent_to_images(latent, "minimax-h3") is None
    assert latent_to_images(None, "default") is None