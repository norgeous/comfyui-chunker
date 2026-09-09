import os
from typing import List, Tuple, Optional
import torch
import torchaudio
import torchaudio.transforms
from functools import reduce
import folder_paths
from comfy_api.latest import AudioInput


def stretch_audio_to_fps(audio_dict: Optional[dict], source_fps: float, target_fps: float) -> Optional[dict]:
    """Time-stretch audio so the same frame count spans source_fps/target_fps x as long (preserves pitch)."""
    if audio_dict is None:
        return None

    waveform = audio_dict["waveform"]
    sample_rate = audio_dict["sample_rate"]

    current_samples = waveform.shape[-1]
    target_samples = int(current_samples * source_fps / target_fps)

    if current_samples == target_samples:
        return audio_dict

    rate = target_fps / source_fps  # TimeStretch: rate > 1 compresses time

    spec = torchaudio.transforms.Spectrogram(n_fft=400, power=None)
    istft = torchaudio.transforms.InverseSpectrogram(n_fft=400)
    stretch = torchaudio.transforms.TimeStretch()

    complex_spec = spec(waveform)
    stretched_spec = stretch(complex_spec, rate)
    stretched_waveform = istft(stretched_spec, length=target_samples)

    return {
        "waveform": stretched_waveform,
        "sample_rate": sample_rate,
    }


def concat_audio(audio1: AudioInput, audio2: AudioInput) -> AudioInput:
    waveform_1 = audio1["waveform"]
    waveform_2 = audio2["waveform"]
    sample_rate_1 = audio1["sample_rate"]
    sample_rate_2 = audio2["sample_rate"]
    if sample_rate_1 != sample_rate_2:
        waveform_2 = torchaudio.functional.resample(waveform_2, sample_rate_2, sample_rate_1)
    concatenated_audio = torch.cat((waveform_1, waveform_2), dim=2)
    return {
        "waveform": concatenated_audio,
        "sample_rate": sample_rate_1,
    }


def concat_audios(audios: List[AudioInput]) -> AudioInput:
    return reduce(lambda a, b: concat_audio(a, b), audios)


def get_next_save_path(filename_prefix: str, extension: str) -> Tuple[str, dict]:
    full_output_folder, filename, counter, subfolder, filename_prefix = (
        folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_temp_directory(),
        )
    )
    file = f"{filename}_{counter:05}_.{extension}"
    full_path = os.path.join(full_output_folder, file)
    frontend_data = {
        "filename": file,
        "subfolder": subfolder,
        "type": "temp",
    }
    return (
        full_path,
        frontend_data,
    )
