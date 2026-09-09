import safetensors.torch
import torch

from lib.utils_latent_combine import concat_chunk_latents


def _save_nested(path, video, audio, latent_type="h3"):
    safetensors.torch.save_file(
        {"latent_0": video, "latent_1": audio},
        path,
        metadata={"type": latent_type},
    )


def _save_single(path, tensor, latent_type="standard"):
    safetensors.torch.save_file({"latent": tensor}, path, metadata={"type": latent_type})


def test_nested_video_and_audio_trim(tmp_path):
    chunk0 = tmp_path / "c0.safetensors"
    chunk1 = tmp_path / "c1.safetensors"
    # chunk 0: 10 video tokens, 5 audio tokens, all 0.0
    _save_nested(
        str(chunk0),
        torch.zeros(1, 24, 10, 4, 4),
        torch.zeros(1, 32, 2, 5),
    )
    # chunk 1: 7 video tokens (first 2 are overlap), 4 audio tokens (first 1 overlap), all 1.0
    _save_nested(
        str(chunk1),
        torch.ones(1, 24, 7, 4, 4),
        torch.ones(1, 32, 2, 4),
    )

    combined, latent_type = concat_chunk_latents(
        [str(chunk0), str(chunk1)],
        video_overlaps=[0, 2],
        audio_overlaps=[0, 1],
    )

    assert latent_type == "h3"
    assert hasattr(combined, "tensors")
    video, audio = combined.tensors
    assert tuple(video.shape) == (1, 24, 15, 4, 4)
    assert tuple(audio.shape) == (1, 32, 2, 8)
    assert bool((video[:, :, :10] == 0).all() and (video[:, :, 10:] == 1).all())
    assert bool((audio[:, :, :, :5] == 0).all() and (audio[:, :, :, 5:] == 1).all())


def test_single_5d_video_trim(tmp_path):
    chunk0 = tmp_path / "c0.safetensors"
    chunk1 = tmp_path / "c1.safetensors"
    _save_single(str(chunk0), torch.zeros(1, 16, 10, 3, 3))
    _save_single(str(chunk1), torch.ones(1, 16, 8, 3, 3))

    combined, latent_type = concat_chunk_latents(
        [str(chunk0), str(chunk1)],
        video_overlaps=[0, 3],
    )

    assert latent_type == "standard"
    assert tuple(combined.shape) == (1, 16, 15, 3, 3)
    assert bool((combined[:, :, :10] == 0).all() and (combined[:, :, 10:] == 1).all())


def test_single_4d_batch_trim(tmp_path):
    chunk0 = tmp_path / "c0.safetensors"
    chunk1 = tmp_path / "c1.safetensors"
    _save_single(str(chunk0), torch.zeros(2, 4, 8, 8))
    _save_single(str(chunk1), torch.ones(3, 4, 8, 8))

    combined, _ = concat_chunk_latents(
        [str(chunk0), str(chunk1)],
        video_overlaps=[0, 1],
    )

    assert tuple(combined.shape) == (4, 4, 8, 8)
    assert bool((combined[:2] == 0).all() and (combined[2:] == 1).all())


def test_first_chunk_no_trim(tmp_path):
    chunk0 = tmp_path / "c0.safetensors"
    _save_single(str(chunk0), torch.zeros(1, 16, 5, 3, 3))

    combined, latent_type = concat_chunk_latents([str(chunk0)], video_overlaps=[7])

    assert latent_type == "standard"
    assert tuple(combined.shape) == (1, 16, 5, 3, 3)


def test_no_paths():
    combined, latent_type = concat_chunk_latents([])
    assert combined is None
    assert latent_type == "standard"