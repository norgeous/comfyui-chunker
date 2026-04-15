from typing import Optional, Tuple

import av
import numpy as np
import torch


def av_load(path: str, overlap_frame_count:int = 0) -> Tuple[Optional[torch.Tensor], Optional[dict]]:
    container = av.open(path)
    container.seek(0)

    images = None
    audio = None
    fps = 30

    if container.streams.video:
        video_stream = container.streams.video[0]
        fps = int(video_stream.average_rate)

        frames = []
        for frame in container.decode(video_stream):
            arr = frame.to_ndarray(format="rgb24")
            arr = arr.astype(np.float32) / 255.0
            frames.append(torch.from_numpy(arr))

        if frames:
            images = torch.stack(frames)

    container.seek(0)

    if container.streams.audio:
        audio_stream = container.streams.audio[0]
        sample_rate = audio_stream.rate
        is_stereo = audio_stream.layout.name == 'stereo'

        audio_frames = []
        for frame in container.decode(audio_stream):
            arr = frame.to_ndarray()

            if is_stereo:
                arr = arr.reshape(2, -1)
            else:
                arr = arr.flatten()

            audio_frames.append(arr)

        if audio_frames:
            audio_data = np.concatenate(audio_frames, axis=-1)

            if is_stereo:
                waveform = torch.from_numpy(audio_data.astype(np.float32) / 32767.0)
            else:
                waveform = torch.from_numpy(audio_data.astype(np.float32) / 32767.0)
                waveform = waveform.unsqueeze(0)

            audio = {
                "waveform": waveform,
                "sample_rate": sample_rate,
            }


    if overlap_frame_count == 0:
        container.close()
        return images, audio

    sr = audio["sample_rate"]
    overlap_sample_count = sr * (overlap_frame_count / fps)
    audio = {
        "waveform": audio[:-overlap_sample_count],
        "sample_rate": sr,
    }

    container.close()av_load(
    return images[:-overlap_frame_count], audio
