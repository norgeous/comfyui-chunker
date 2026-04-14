import math
from enum import Enum
from fractions import Fraction
from typing import List, Tuple, Callable

import av
import numpy as np


class BlendMode(Enum):
    EQUAL_POWER = "equal_power"
    LINEAR = "linear"
    EASE_IN_OUT = "ease_in_out"
    NEWER_ONLY = "newer_only"
    OLDER_ONLY = "older_only"


BlendHandler = Callable[[float], float]

blend_handlers: dict[BlendMode, BlendHandler] = {
    BlendMode.EQUAL_POWER: lambda p: math.sin(p * math.pi / 2),
    BlendMode.LINEAR: lambda p: p,
    BlendMode.EASE_IN_OUT: lambda p: p * p * (3 - 2 * p),
    BlendMode.NEWER_ONLY: lambda p: 1,
    BlendMode.OLDER_ONLY: lambda p: 0,
}


def get_blend_factor(mode: BlendMode, percent: float) -> float:
    return blend_handlers[mode](percent)


def load_video(path: str) -> Tuple[List[av.VideoFrame], np.ndarray, int, int]:
    inp: av.container.input.InputContainer = av.open(path)
    inp.seek(0)
    
    in_video: av.stream.Stream = inp.streams.video[0]
    fps: int = int(in_video.average_rate)

    audio_data: np.ndarray = None  # type: ignore[assignment]
    sample_rate: int = 0
    if len(inp.streams.audio) > 0:
        in_audio: av.stream.Stream = inp.streams.audio[0]
        sample_rate = in_audio.rate
        audio_frames: List[np.ndarray] = [frame.to_ndarray() for frame in inp.decode(in_audio)]
        if audio_frames:
            audio_data = np.concatenate(audio_frames, axis=-1).flatten()

    inp.seek(0)
    frames: List[av.VideoFrame] = list(inp.decode(in_video))

    inp.close()  # type: ignore[union-attr]
    return frames, audio_data, fps, sample_rate


def create_overlap_frames(
    end_frames: List[av.VideoFrame],
    start_frames: List[av.VideoFrame],
    overlap_count: int,
    blend_mode: BlendMode
) -> List[av.VideoFrame]:
    blended: List[av.VideoFrame] = []
    for j in range(overlap_count):
        alpha: float = get_blend_factor(blend_mode, (j + 1) / (overlap_count + 1))
        arr_a: np.ndarray = end_frames[j].to_ndarray(format="rgb24").astype(float)
        arr_b: np.ndarray = start_frames[j].to_ndarray(format="rgb24").astype(float)
        blended_arr: np.ndarray = (arr_a * (1 - alpha) + arr_b * alpha).astype(np.uint8)
        blended.append(av.VideoFrame.from_ndarray(blended_arr, format="rgb24"))
    return blended


def create_overlap_audio(
    end_samples: np.ndarray,
    start_samples: np.ndarray,
    blend_mode: BlendMode
) -> np.ndarray:
    progress: np.ndarray = np.linspace(0, 1, len(end_samples))
    alpha: np.ndarray = np.array([get_blend_factor(blend_mode, p) for p in progress])
    return (end_samples * (1 - alpha) + start_samples * alpha).astype(np.int16)


def trim_frames(frames: List[av.VideoFrame], remove_start: int, remove_end: int) -> List[av.VideoFrame]:
    if remove_start > 0 and remove_end > 0:
        return frames[remove_start:-remove_end] if len(frames) > remove_start + remove_end else frames
    elif remove_start > 0:
        return frames[remove_start:]
    elif remove_end > 0:
        return frames[:-remove_end]
    return frames


def trim_audio(samples: np.ndarray, remove_start: int, remove_end: int) -> np.ndarray:
    if remove_start > 0 and remove_end > 0:
        return samples[remove_start:-remove_end] if len(samples) > remove_start + remove_end else samples
    elif remove_start > 0:
        return samples[remove_start:]
    elif remove_end > 0:
        return samples[:-remove_end]
    return samples


def combine(
    paths: List[str],
    output_path: str,
    overlap_frame_count: int = 10,
    video_blend_mode: BlendMode = BlendMode.LINEAR,
    audio_blend_mode: BlendMode = BlendMode.EQUAL_POWER
) -> None:
    sources: List[dict[str, List[av.VideoFrame] | np.ndarray | int]] = []
    for path in paths:
        frames: List[av.VideoFrame]
        audio: np.ndarray
        fps: int
        sr: int
        frames, audio, fps, sr = load_video(path)
        sources.append({"frames": frames, "audio": audio, "fps": fps, "sample_rate": sr})

    overlaps: List[dict[str, List[av.VideoFrame] | np.ndarray | None]] = []
    for i in range(len(sources) - 1):
        a: dict[str, List[av.VideoFrame] | np.ndarray | int] = sources[i]
        b: dict[str, List[av.VideoFrame] | np.ndarray | int] = sources[i + 1]

        overlap_frames: List[av.VideoFrame] = []
        overlap_audio: np.ndarray = None  # type: ignore[assignment]
        
        if overlap_frame_count > 0:
            end_frames: List[av.VideoFrame] = a["frames"][-overlap_frame_count:]
            start_frames: List[av.VideoFrame] = b["frames"][:overlap_frame_count]
            overlap_frames = create_overlap_frames(end_frames, start_frames, overlap_frame_count, video_blend_mode)

            if a["audio"] is not None and b["audio"] is not None:
                a_1d: np.ndarray = a["audio"].flatten() if a["audio"].ndim > 1 else a["audio"]
                b_1d: np.ndarray = b["audio"].flatten() if b["audio"].ndim > 1 else b["audio"]
                overlap_samples: int = overlap_frame_count * (sr // fps)
                end_audio: np.ndarray = a_1d[-overlap_samples:]
                start_audio: np.ndarray = b_1d[:overlap_samples]
                overlap_audio = create_overlap_audio(end_audio, start_audio, audio_blend_mode)

        overlaps.append({"frames": overlap_frames, "audio": overlap_audio})

    final_frames: List[av.VideoFrame] = []
    final_audio: List[np.ndarray] = []

    for idx, src in enumerate(sources):
        remove_start: int = overlap_frame_count if idx > 0 else 0
        remove_end: int = overlap_frame_count if idx < len(sources) - 1 else 0

        trimmed: List[av.VideoFrame] = trim_frames(src["frames"], remove_start, remove_end)
        final_frames.extend(trimmed)

        if src["audio"] is not None:
            audio_1d: np.ndarray = src["audio"].flatten() if src["audio"].ndim > 1 else src["audio"]
            trimmed_audio: np.ndarray = trim_audio(audio_1d, remove_start * (sr // fps), remove_end * (sr // fps))
            final_audio.append(trimmed_audio)

        if idx < len(overlaps):
            final_frames.extend(overlaps[idx]["frames"])
            if overlaps[idx]["audio"] is not None:
                final_audio.append(overlaps[idx]["audio"])
    output: av.container.output.OutputContainer = av.open(output_path, mode='w')

    out_video: av.stream.Stream = output.add_stream('h264', rate=fps)
    out_video.width = sources[0]["frames"][0].width
    out_video.height = sources[0]["frames"][0].height
    out_video.pix_fmt = 'yuv420p'
    out_video.options = {'preset': 'slow', 'crf': '10', 'gop_size': '1', 'keyint_min': '1'}

    out_audio: av.stream.Stream = None  # type: ignore[assignment]
    if final_audio:
        out_audio = output.add_stream('pcm_s16le', rate=sr)
        out_audio.layout = 'mono'

    for i, frame in enumerate(final_frames):
        frame = frame.reformat(format='yuv420p')
        frame.pts = i
        frame.time_base = Fraction(1, fps)
        for pkt in out_video.encode(frame):
            output.mux(pkt)

    for pkt in out_video.encode():
        output.mux(pkt)

    if out_audio and final_audio:
        audio_concat: np.ndarray = np.concatenate(final_audio)
        audio_frame: av.audio.AudioFrame = av.AudioFrame.from_ndarray(audio_concat.reshape(1, -1), format='s16', layout='mono')
        audio_frame.rate = sr
        audio_frame.pts = 0
        audio_frame.time_base = Fraction(1, sr)

        for pkt in out_audio.encode(audio_frame):
            output.mux(pkt)

        for pkt in out_audio.encode():
            output.mux(pkt)

    output.close()
