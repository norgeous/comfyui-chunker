import math
from enum import Enum
from typing import List
import numpy as np
import torch
from .av_load import av_load
from .av_save import av_save, Profile


class BlendMode(Enum):
    EQUAL_POWER = "equal_power"
    LINEAR = "linear"
    EASE_IN_OUT = "ease_in_out"
    NEWER_ONLY = "newer_only"
    OLDER_ONLY = "older_only"


blend_handlers = {
    BlendMode.EQUAL_POWER: lambda p: math.sin(p * math.pi / 2),
    BlendMode.LINEAR: lambda p: p,
    BlendMode.EASE_IN_OUT: lambda p: p * p * (3 - 2 * p),
    BlendMode.NEWER_ONLY: lambda p: 1,
    BlendMode.OLDER_ONLY: lambda p: 0,
}


def get_blend_factor(mode: BlendMode, percent: float) -> float:
    return blend_handlers[mode](percent)


def create_overlap_frames(
    end_frames: torch.Tensor,
    start_frames: torch.Tensor,
    overlap_count: int,
    blend_mode: BlendMode,
) -> torch.Tensor:
    blended = []
    for j in range(overlap_count):
        alpha = get_blend_factor(blend_mode, (j + 1) / (overlap_count + 1))
        arr_a = end_frames[j].numpy().astype(float)
        arr_b = start_frames[j].numpy().astype(float)
        arr_a = (arr_a * 255).astype(np.uint8)
        arr_b = (arr_b * 255).astype(np.uint8)
        blended_arr = (arr_a * (1 - alpha) + arr_b * alpha).astype(np.uint8)
        blended_arr = blended_arr.astype(np.float32) / 255.0
        blended.append(torch.from_numpy(blended_arr))
    return torch.stack(blended)


def create_overlap_audio(
    end_samples: np.ndarray,
    start_samples: np.ndarray,
    blend_mode: BlendMode,
) -> np.ndarray:
    progress = np.linspace(0, 1, len(end_samples))
    alpha = np.array([get_blend_factor(blend_mode, p) for p in progress])
    blended = end_samples * (1 - alpha) + start_samples * alpha
    return blended.astype(np.float32)


def av_combine(
    paths: List[str],
    output_path: str,
    overlap_frame_count: int = 10,
    video_blend_mode: BlendMode = BlendMode.LINEAR,
    audio_blend_mode: BlendMode = BlendMode.EQUAL_POWER,
    profile: Profile = Profile.HQ,
) -> str:
    sources = []
    for path in paths:
        images, audio, fps = av_load(path)
        sources.append({"images": images, "audio": audio, "fps": fps})

    if len(sources) == 0:
        raise ValueError("No sources provided")

    fps = sources[0]["fps"]

    overlaps = []
    for i in range(len(sources) - 1):
        a = sources[i]
        b = sources[i + 1]

        overlap_frames = []
        overlap_audio = None

        if overlap_frame_count > 0 and a["images"] is not None and b["images"] is not None:
            end_frames = a["images"][-overlap_frame_count:]
            start_frames = b["images"][:overlap_frame_count]
            overlap_frames = create_overlap_frames(end_frames, start_frames, overlap_frame_count, video_blend_mode)

        if a["audio"] is not None and b["audio"] is not None:
            is_stereo = a["audio"]["waveform"].shape[1] == 2
            sr = a["audio"]["sample_rate"]
            source_fps = a["fps"]  # Use actual FPS from video

            if is_stereo:
                a_left = a["audio"]["waveform"][0, 0].numpy()
                b_left = b["audio"]["waveform"][0, 0].numpy()
                a_right = a["audio"]["waveform"][0, 1].numpy()
                b_right = b["audio"]["waveform"][0, 1].numpy()
            else:
                a_left = a["audio"]["waveform"][0, 0].numpy()
                b_left = b["audio"]["waveform"][0, 0].numpy()
                a_right = None
                b_right = None

            overlap_samples = overlap_frame_count * (sr // source_fps)
            end_audio = a_left[-overlap_samples:]
            start_audio = b_left[:overlap_samples]

            if len(start_audio) > 0 and overlap_frame_count > 0:
                overlap_audio = create_overlap_audio(end_audio, start_audio, audio_blend_mode)

                if is_stereo:
                    end_audio_r = a_right[-overlap_samples:]
                    start_audio_r = b_right[:overlap_samples]
                    overlap_audio_r = create_overlap_audio(end_audio_r, start_audio_r, audio_blend_mode)
                    overlap_audio = np.stack([overlap_audio, overlap_audio_r], axis=0)
            else:
                overlap_audio = None

        overlaps.append({"frames": overlap_frames, "audio": overlap_audio})

    final_frames = []
    final_audio = []

    for idx, src in enumerate(sources):
        remove_start = overlap_frame_count if idx > 0 else 0
        remove_end = overlap_frame_count if idx < len(sources) - 1 else 0

        if src["images"] is not None:
            if remove_start > 0 or remove_end > 0:
                trimmed = src["images"][remove_start:-remove_end] if remove_end > 0 else src["images"][remove_start:]
            else:
                trimmed = src["images"]
            final_frames.append(trimmed)

        if src["audio"] is not None:
            sr = src["audio"]["sample_rate"]
            source_fps = src["fps"]  # Use actual FPS from video
            is_stereo = src["audio"]["waveform"].shape[1] == 2

            if is_stereo:
                left = src["audio"]["waveform"][0, 0].numpy()
                right = src["audio"]["waveform"][0, 1].numpy()
                trim_start = remove_start * (sr // source_fps)
                trim_end = remove_end * (sr // source_fps)
                trimmed_left = left[trim_start:-trim_end] if trim_end > 0 else left[trim_start:]
                trimmed_right = right[trim_start:-trim_end] if trim_end > 0 else right[trim_start:]
                trimmed_audio = np.stack([trimmed_left, trimmed_right], axis=0)
            else:
                audio_1d = src["audio"]["waveform"][0, 0].numpy()
                trimmed_audio = audio_1d[remove_start * (sr // source_fps):-remove_end * (sr // source_fps)] if remove_end > 0 else audio_1d[remove_start * (sr // source_fps):]

            final_audio.append(trimmed_audio)

        if idx < len(overlaps):
            if overlaps[idx]["frames"] is not None and len(overlaps[idx]["frames"]) > 0:
                final_frames.append(overlaps[idx]["frames"])
            if overlaps[idx]["audio"] is not None:
                final_audio.append(overlaps[idx]["audio"])

    final_images = torch.cat(final_frames, dim=0) if final_frames else None

    final_audio_dict = None
    if final_audio:
        sr = sources[0]["audio"]["sample_rate"]

        shapes = [a.shape for a in final_audio]
        dims = [a.ndim for a in final_audio]
        if len(set(dims)) > 1:
            raise ValueError(f"Audio dimension mismatch: cannot combine arrays with different shapes {shapes}")

        is_stereo = final_audio[0].ndim == 2 and final_audio[0].shape[0] == 2
        if is_stereo:
            audio_concat = np.concatenate(final_audio, axis=1)
            final_audio_dict = {
                "waveform": torch.from_numpy(audio_concat).unsqueeze(0).float(),
                "sample_rate": sr,
            }
        else:
            audio_concat = np.concatenate(final_audio, axis=0).reshape(1, -1)
            final_audio_dict = {
                "waveform": torch.from_numpy(audio_concat).float(),
                "sample_rate": sr,
            }

    return av_save(images=final_images, audio=final_audio_dict, output_path=output_path, fps=fps, profile=profile)
