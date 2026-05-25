import os
import folder_paths
from functools import reduce
from comfy_extras.nodes_audio import match_audio_sample_rates
import torch


def concat_audio(audio1, audio2):
    waveform_1 = audio1["waveform"]
    waveform_2 = audio2["waveform"]
    sample_rate_1 = audio1["sample_rate"]
    sample_rate_2 = audio2["sample_rate"]
    waveform_1, waveform_2, output_sample_rate = match_audio_sample_rates(
        waveform_1, sample_rate_1, waveform_2, sample_rate_2)
    concatenated_audio = torch.cat((waveform_1, waveform_2), dim=2)
    return {
        "waveform": concatenated_audio,
        "sample_rate": output_sample_rate,
    }


def concat_audios(audios):
    return reduce(lambda a, b: concat_audio(a, b), audios)


def get_next_save_path(filename_prefix, extension):
    full_output_folder, filename, counter, subfolder, filename_prefix = (
        folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_temp_directory(),
        )
    )
    file = f"{filename}_{counter:05}_.{extension}"
    full_path = os.path.join(full_output_folder, file)
    return (
        full_path,
        {
            "filename": file,
            "subfolder": subfolder,
            "type": "temp",
        },
    )
