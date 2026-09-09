import torch

from lib.av_encode import encode_video, encode_audio, pack_av_latent, enforce_stereo


class FakeVideoVAE:
    def encode(self, images):
        n, h, w, c = images.shape
        t = (n + 3) // 4
        return torch.zeros([1, 24, t, h // 16, w // 16])


class FakeAudioVAE:
    audio_sample_rate = 32000

    def __init__(self):
        self.last_input = None

    def encode(self, samples):
        # samples: [B, L, C]
        self.last_input = samples
        b, l, c = samples.shape
        return torch.zeros([b, 32, 2, l // 800])


def test_enforce_stereo_mono():
    mono = torch.rand([1, 1, 1000])
    stereo = enforce_stereo(mono)
    assert stereo.shape == (1, 2, 1000)
    assert torch.allclose(stereo[:, 0], mono[:, 0])
    assert torch.allclose(stereo[:, 1], mono[:, 0])


def test_enforce_stereo_passthrough():
    stereo = torch.rand([2, 2, 1000])
    assert enforce_stereo(stereo) is stereo


def test_encode_audio_resamples_and_shapes():
    vae = FakeAudioVAE()
    waveform = torch.rand([1, 2, 48000]) * 2 - 1
    audio = {"waveform": waveform, "sample_rate": 48000}
    latent = encode_audio(vae, audio)
    assert latent.shape == (1, 32, 2, 40)
    assert vae.last_input.shape[0] == 1
    assert vae.last_input.shape[2] == 2
    assert vae.last_input.shape[1] == 32000


def test_encode_audio_skips_resample_when_matching():
    vae = FakeAudioVAE()
    waveform = torch.rand([1, 2, 32000]) * 2 - 1
    latent = encode_audio(vae, {"waveform": waveform, "sample_rate": 32000})
    assert latent.shape == (1, 32, 2, 40)
    assert vae.last_input.shape[1] == 32000


def test_encode_audio_mono_becomes_stereo():
    vae = FakeAudioVAE()
    waveform = torch.rand([1, 1, 32000]) * 2 - 1
    latent = encode_audio(vae, {"waveform": waveform, "sample_rate": 32000})
    assert latent.shape == (1, 32, 2, 40)
    assert vae.last_input.shape[2] == 2


def test_encode_video_shape():
    vae = FakeVideoVAE()
    images = torch.rand([39, 64, 64, 3])
    latent = encode_video(vae, images)
    assert latent.shape == (1, 24, 10, 4, 4)


def test_pack_av_latent_both_streams():
    video = torch.zeros([1, 24, 10, 4, 4])
    audio = torch.zeros([1, 32, 2, 40])
    nt = pack_av_latent(video, audio)
    assert hasattr(nt, "tensors")
    assert len(nt.tensors) == 2
    assert nt.tensors[0].shape == video.shape
    assert nt.tensors[1].shape == audio.shape


def test_pack_av_latent_video_only():
    video = torch.zeros([1, 24, 10, 4, 4])
    nt = pack_av_latent(video)
    assert not hasattr(nt, "tensors")
    assert nt.shape == video.shape