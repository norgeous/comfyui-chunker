import torch
import av
import folder_paths
from comfy_extras.nodes_audio import match_audio_sample_rates
from functools import reduce

# from https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_audio.py

def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format."""
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    elif wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")

def load_audio(filepath):
    audio_path = folder_paths.get_annotated_filepath(filepath)
    with av.open(audio_path) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream found in the file.")

        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        n_channels = stream.channels

        frames = []
        length = 0
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()

            frames.append(buf)
            length += buf.shape[1]

        if not frames:
            raise ValueError("No audio frames decoded.")

        wav = torch.cat(frames, dim=1)
        wav = f32_pcm(wav)
        audio = {
            "waveform": wav.unsqueeze(0),
            "sample_rate": sr,
        }
        return audio

def concat_audio(audio1, audio2):
    waveform_1 = audio1["waveform"]
    waveform_2 = audio2["waveform"]
    sample_rate_1 = audio1["sample_rate"]
    sample_rate_2 = audio2["sample_rate"]
    if waveform_1.shape[1] == 1:
        waveform_1 = waveform_1.repeat(1, 2, 1) # Convert mono to stereo
    if waveform_2.shape[1] == 1:
        waveform_2 = waveform_2.repeat(1, 2, 1) # Convert mono to stereo
    waveform_1, waveform_2, output_sample_rate = match_audio_sample_rates(waveform_1, sample_rate_1, waveform_2, sample_rate_2)
    concatenated_audio = torch.cat((waveform_1, waveform_2), dim=2)
    return {
        "waveform": concatenated_audio,
        "sample_rate": output_sample_rate,
    }

def concat_audios(audios):
    return reduce(lambda a, b: concat_audio(a, b), audios)
