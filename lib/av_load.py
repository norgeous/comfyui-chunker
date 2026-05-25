from typing import Optional, Tuple
import av
import numpy as np
import torch
import math


def av_load(path: str,
            overlap_frame_count: int = 0) -> Tuple[Optional[torch.Tensor],
                                                   Optional[torch.Tensor],
                                                   Optional[dict],
                                                   int]:
    container = av.open(path)
    container.seek(0)

    images = None
    masks = None
    audio = None
    fps = 30

    if container.streams.video:
        video_stream = container.streams.video[0]
        video_stream.thread_count = 0
        video_stream.thread_type = "AUTO"
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

    if len(container.streams.video) > 1 and images is not None:
        mask_stream = container.streams.video[1]
        mask_stream.thread_count = 0
        mask_stream.thread_type = "AUTO"

        mask_frames = []
        for frame in container.decode(mask_stream):
            arr = frame.to_ndarray(format="gray")
            arr = arr.astype(np.float32) / 255.0
            mask_frames.append(torch.from_numpy(arr))

        if mask_frames:
            masks = torch.stack(mask_frames)[vstart:vend]

        container.seek(0)

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

            if is_stereo:
                if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[0] == 1):
                    # Interleaved format (s16, etc.) - shape (samples*2,) or
                    # (1, samples*2)
                    flat = arr.flatten()
                    arr = np.stack([flat[::2], flat[1::2]], axis=0)[
                        np.newaxis, :, :]
                else:
                    # Planar format (fltp) - shape (channels, samples)
                    arr = arr[np.newaxis, :, :]
            else:
                arr = arr.flatten()

            audio_frames.append(arr)

        if audio_frames:
            if is_stereo:
                audio_data = np.concatenate(audio_frames, axis=2)
            else:
                audio_data = np.concatenate(audio_frames)

            if np.issubdtype(audio_data.dtype, np.floating):
                waveform = torch.from_numpy(audio_data.astype(np.float32))
            else:
                waveform = torch.from_numpy(
                    audio_data.astype(
                        np.float32) /
                    np.iinfo(
                        audio_data.dtype).max)

            if not is_stereo and waveform.dim() == 1:
                waveform = waveform.reshape(1, -1)
            if is_stereo:
                waveform = waveform[:, :, astart:aend]
            else:
                waveform = waveform.unsqueeze(0)
                waveform = waveform[:, :, astart:aend]

            audio = {
                "waveform": waveform,
                "sample_rate": sample_rate,
            }

    container.close()
    return images, masks, audio, fps
