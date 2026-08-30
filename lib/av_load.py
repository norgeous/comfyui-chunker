from typing import Optional, Tuple
import av
import numpy as np
import torch
import math


def _frame_count(container) -> Tuple[Optional[int], bool]:
    """Return (frame_count, is_exact). is_exact=False when derived from duration estimate."""
    if not container.streams.video:
        return None, False
    vs = container.streams.video[0]
    if vs.frames and vs.frames > 0:
        return int(vs.frames), True
    if vs.average_rate and container.duration:
        est = int(float(container.duration / av.time_base) * float(vs.average_rate))
        if est > 0:
            return est, False
    n = 0
    for _ in container.decode(vs):
        n += 1
    if n > 0:
        return n, True
    return None, False


def av_load(path: str, start: int = 0, end: Optional[int] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[dict], int]:
    container = av.open(path)

    need_probe = (start is not None and start < 0) or (end is not None and end < 0)
    total = None
    is_exact = False
    if need_probe:
        total, is_exact = _frame_count(container)

    slack = 0 if is_exact else 8
    negative_start_target = None
    if start is not None and start < 0:
        negative_start_target = -start
        start = max(0, (total or 0) + start - slack)
    if end is not None and end < 0:
        end = max(0, (total or 0) + end)

    images = None
    masks = None
    audio = None
    fps = 30

    if container.streams.video:
        video_stream = container.streams.video[0]
        video_stream.thread_count = 0
        video_stream.thread_type = "AUTO"
        fps = int(video_stream.average_rate)
        rate = float(video_stream.average_rate)
        half = 0.5 / rate

        start_t = start / rate
        end_t = end / rate if end is not None else None

        if start > 0:
            tb = float(video_stream.time_base) if video_stream.time_base else 1.0 / rate
            container.seek(max(0, int(start_t / tb)), stream=video_stream, backward=True)
        else:
            container.seek(0)

        frames = []

        for frame in container.decode(video_stream):
            if frame.pts is not None:
                t = float(frame.pts * video_stream.time_base)
                if t < start_t - half:
                    continue
                if end_t is not None and t >= end_t - half:
                    break
            arr = frame.to_ndarray(format="rgb24")
            arr = arr.astype(np.float32) / 255.0
            frames.append(torch.from_numpy(arr))

        if frames:
            images = torch.stack(frames)

    container.seek(0)

    if len(container.streams.video) > 1 and images is not None:
        mask_stream = container.streams.video[1]
        mask_stream.thread_count = 0
        mask_stream.thread_type = "AUTO"

        if start > 0:
            tb = float(mask_stream.time_base) if mask_stream.time_base else 1.0 / rate
            container.seek(max(0, int(start_t / tb)), stream=mask_stream, backward=True)
        else:
            container.seek(0)

        mask_frames = []
        for frame in container.decode(mask_stream):
            if frame.pts is not None:
                t = float(frame.pts * mask_stream.time_base)
                if t < start_t - half:
                    continue
                if end_t is not None and t >= end_t - half:
                    break
            arr = frame.to_ndarray(format="gray")
            arr = arr.astype(np.float32) / 255.0
            mask_frames.append(torch.from_numpy(arr))

        if mask_frames:
            masks = torch.stack(mask_frames)

    container.seek(0)

    if container.streams.audio:
        audio_stream = container.streams.audio[0]
        sample_rate = audio_stream.rate
        is_stereo = audio_stream.layout.name == "stereo"

        samples_per_frame = math.floor(sample_rate / fps)

        astart = start * samples_per_frame
        aend = end * samples_per_frame if end is not None else None

        audio_frames = []
        for frame in container.decode(audio_stream):
            arr = frame.to_ndarray()

            if is_stereo:
                if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[0] == 1):
                    flat = arr.flatten()
                    arr = np.stack([flat[::2], flat[1::2]], axis=0)[
                        np.newaxis, :, :]
                else:
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

    # Post-decode trim when negative start was requested
    if negative_start_target is not None:
        target = negative_start_target
        if images is not None and len(images) > target:
            images = images[-target:]
        if masks is not None and len(masks) > target:
            masks = masks[-target:]
        if audio is not None:
            spb = samples_per_frame
            audio["waveform"] = audio["waveform"][:, :, -target * spb:]

    container.close()
    return images, masks, audio, fps
