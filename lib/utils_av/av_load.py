from typing import Optional, Tuple

import av
import numpy as np
import torch
import math


def av_load(path: str, overlap_frame_count:int = 0) -> Tuple[Optional[torch.Tensor], Optional[dict], int]:
    container = av.open(path)
    container.seek(0)

    images = None
    audio = None
    fps = 30

    if container.streams.video:
        video_stream = container.streams.video[0]
        fps = int(video_stream.average_rate)
           
        vstart = None
        vend = None
        if overlap_frame_count > 0:
            vend = overlap_frame_count
        if overlap_frame_count < 0:
            vstart = overlap_frame_count
      
        frames = []
        for frame in container.decode(video_stream):
            arr = frame.to_ndarray(format="rgb24")
            arr = arr.astype(np.float32) / 255.0
            frames.append(torch.from_numpy(arr))

        if frames:
            images = torch.stack(frames)[vstart:vend]

    container.seek(0)

    if container.streams.audio:
        audio_stream = container.streams.audio[0]
        sample_rate = audio_stream.rate
        is_stereo = audio_stream.layout.name == 'stereo'

        samples_per_frame = math.floor(sample_rate / fps)

        astart = None
        aend = None
        if overlap_frame_count > 0:
            aend = overlap_frame_count * samples_per_frame
        if overlap_frame_count < 0:
            astart = overlap_frame_count * samples_per_frame

        audio_frames = []
        for frame in container.decode(audio_stream):
            arr = frame.to_ndarray()
            is_planar = frame.format.is_planar

            if is_stereo:
                if is_planar:
                    # Planar stereo: already (2, num_samples)
                    pass
                else:
                    # Packed stereo: (1, num_samples * 2) → (2, num_samples)
                    arr = arr.reshape(2, -1)
            else:
                # Mono: always flatten (1, num_samples) → (num_samples,)
                arr = arr.flatten()

            audio_frames.append(arr)

        if audio_frames:
            audio_data = np.concatenate(audio_frames, axis=-1)

            waveform = torch.from_numpy(audio_data.astype(np.float32) / 32767.0)
            # Add batch dimension: (channels, samples) → (1, channels, samples)
            # For mono: (samples,) → (1, samples) → (1, 1, samples)
            # For stereo: (2, samples) → (1, 2, samples)
            if waveform.dim() == 1:
                # Mono audio: add channel dimension first
                waveform = waveform.unsqueeze(0)  # (samples,) → (1, samples)
            waveform = waveform.unsqueeze(0)  # Add batch dimension
            # Apply slicing: (1, channels, samples) → (1, channels, start:end)
            waveform = waveform[:, :, astart:aend]
          
            audio = {
                "waveform": waveform,
                "sample_rate": sample_rate,
            }

    container.close()
    return images, audio, fps

