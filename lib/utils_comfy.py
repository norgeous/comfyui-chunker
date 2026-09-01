import os
from typing import List, Tuple
import torch
import torchaudio
from functools import reduce
import folder_paths
from comfy_api.latest import AudioInput


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
