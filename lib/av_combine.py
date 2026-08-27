import math
from enum import Enum
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from .av_load import av_load
from .av_save import av_save, Profile

Source = Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[dict], float]


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
    BlendMode.NEWER_ONLY: lambda _p: 1,
    BlendMode.OLDER_ONLY: lambda _p: 0,
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
    inputs: List[Union[str, Source]],
    filename_prefix: str = "output",
    overlap_frame_count: int = 10,
    video_blend_mode: BlendMode = BlendMode.LINEAR,
    audio_blend_mode: BlendMode = BlendMode.EQUAL_POWER,
    profile: Profile = Profile.HQ,
    need_images: bool = True,
    need_masks: bool = True,
    need_audio: bool = True,
) -> Tuple[str, dict, Optional[torch.Tensor], Optional[torch.Tensor], Optional[dict]]:
    def _load(item):
        if isinstance(item, str):
            return av_load(item)
        return item

    if len(inputs) == 0:
        raise ValueError("No sources provided")

    # Sliding window: load at most two chunks at a time so each chunk's
    # tensors are released from memory as soon as they have been consumed.
    prev_images, prev_masks, prev_audio, prev_fps = _load(inputs[0])
    fps = prev_fps
    audio_sr = prev_audio["sample_rate"] if prev_audio is not None else None

    final_frames = []
    final_masks = []
    final_audio = []

    for i in range(1, len(inputs)):
        curr_images, curr_masks, curr_audio, curr_fps = _load(inputs[i])

        overlap_frames = []
        overlap_masks = []
        overlap_audio = None

        if (overlap_frame_count > 0
                and prev_images is not None
                and curr_images is not None):
            end_frames = prev_images[-overlap_frame_count:]
            start_frames = curr_images[:overlap_frame_count]
            overlap_frames = create_overlap_frames(
                end_frames, start_frames, overlap_frame_count,
                video_blend_mode)

        if (overlap_frame_count > 0
                and prev_masks is not None
                and curr_masks is not None):
            end_masks = prev_masks[-overlap_frame_count:]
            start_masks = curr_masks[:overlap_frame_count]
            overlap_masks = create_overlap_frames(
                end_masks, start_masks, overlap_frame_count, video_blend_mode)

        if prev_audio is not None and curr_audio is not None:
            is_stereo = prev_audio["waveform"].shape[1] == 2
            sr = prev_audio["sample_rate"]
            source_fps = prev_fps

            if is_stereo:
                a_left = prev_audio["waveform"][0, 0].numpy()
                b_left = curr_audio["waveform"][0, 0].numpy()
                a_right = prev_audio["waveform"][0, 1].numpy()
                b_right = curr_audio["waveform"][0, 1].numpy()
            else:
                a_left = prev_audio["waveform"][0, 0].numpy()
                b_left = curr_audio["waveform"][0, 0].numpy()
                a_right = None
                b_right = None

            samples_per_frame = int(sr // source_fps)
            overlap_samples = overlap_frame_count * samples_per_frame
            end_audio = a_left[-overlap_samples:]
            start_audio = b_left[:overlap_samples]

            if len(start_audio) > 0 and overlap_frame_count > 0:
                overlap_audio = create_overlap_audio(
                    end_audio, start_audio, audio_blend_mode)

                if is_stereo:
                    end_audio_r = a_right[-overlap_samples:]
                    start_audio_r = b_right[:overlap_samples]
                    overlap_audio_r = create_overlap_audio(
                        end_audio_r, start_audio_r, audio_blend_mode)
                    overlap_audio = np.stack(
                        [overlap_audio, overlap_audio_r], axis=0)

        # prev is chunk (i - 1); trim and clone it, then release it.
        remove_start = 0 if i == 1 else overlap_frame_count
        remove_end = overlap_frame_count

        if prev_images is not None:
            if remove_start > 0 or remove_end > 0:
                trimmed = (
                    prev_images[remove_start:-remove_end]
                    if remove_end > 0
                    else prev_images[remove_start:])
            else:
                trimmed = prev_images
            final_frames.append(trimmed.clone())

        if prev_masks is not None:
            if remove_start > 0 or remove_end > 0:
                trimmed = (
                    prev_masks[remove_start:-remove_end]
                    if remove_end > 0
                    else prev_masks[remove_start:])
            else:
                trimmed = prev_masks
            final_masks.append(trimmed.clone())

        if prev_audio is not None:
            sr = prev_audio["sample_rate"]
            source_fps = prev_fps
            is_stereo = prev_audio["waveform"].shape[1] == 2
            samples_per_frame = int(sr // source_fps)

            if is_stereo:
                left = prev_audio["waveform"][0, 0].numpy()
                right = prev_audio["waveform"][0, 1].numpy()
                trim_start = remove_start * samples_per_frame
                trim_end = remove_end * samples_per_frame
                trimmed_left = (
                    left[trim_start:-trim_end]
                    if trim_end > 0
                    else left[trim_start:])
                trimmed_right = (
                    right[trim_start:-trim_end]
                    if trim_end > 0
                    else right[trim_start:])
                trimmed_audio = np.stack([trimmed_left, trimmed_right], axis=0)
            else:
                audio_1d = prev_audio["waveform"][0, 0].numpy()
                trimmed_audio = (
                    audio_1d[slice(
                        remove_start * samples_per_frame,
                        -remove_end * samples_per_frame,
                    )]
                    if trim_end > 0
                    else audio_1d[
                        remove_start * samples_per_frame:])
            final_audio.append(trimmed_audio)

        if len(overlap_frames) > 0:
            final_frames.append(overlap_frames)
        if len(overlap_masks) > 0:
            final_masks.append(overlap_masks)
        if overlap_audio is not None:
            final_audio.append(overlap_audio)

        # Release the previous chunk's tensors before loading the next one.
        prev_images = None
        prev_masks = None
        prev_audio = None

        prev_images = curr_images
        prev_masks = curr_masks
        prev_audio = curr_audio

    # Final chunk: only its start overlap is trimmed.
    remove_start = overlap_frame_count if len(inputs) > 1 else 0
    remove_end = 0

    if prev_images is not None:
        if remove_start > 0 or remove_end > 0:
            trimmed = (
                prev_images[remove_start:-remove_end]
                if remove_end > 0
                else prev_images[remove_start:])
        else:
            trimmed = prev_images
        final_frames.append(trimmed.clone())

    if prev_masks is not None:
        if remove_start > 0 or remove_end > 0:
            trimmed = (
                prev_masks[remove_start:-remove_end]
                if remove_end > 0
                else prev_masks[remove_start:])
        else:
            trimmed = prev_masks
        final_masks.append(trimmed.clone())

    if prev_audio is not None:
        sr = prev_audio["sample_rate"]
        source_fps = prev_fps
        is_stereo = prev_audio["waveform"].shape[1] == 2
        samples_per_frame = int(sr // source_fps)

        if is_stereo:
            left = prev_audio["waveform"][0, 0].numpy()
            right = prev_audio["waveform"][0, 1].numpy()
            trim_start = remove_start * samples_per_frame
            trim_end = remove_end * samples_per_frame
            trimmed_left = (
                left[trim_start:-trim_end]
                if trim_end > 0
                else left[trim_start:])
            trimmed_right = (
                right[trim_start:-trim_end]
                if trim_end > 0
                else right[trim_start:])
            trimmed_audio = np.stack([trimmed_left, trimmed_right], axis=0)
        else:
            audio_1d = prev_audio["waveform"][0, 0].numpy()
            trimmed_audio = (
                audio_1d[slice(
                    remove_start * samples_per_frame,
                    -remove_end * samples_per_frame,
                )]
                if trim_end > 0
                else audio_1d[
                    remove_start * samples_per_frame:])
        final_audio.append(trimmed_audio)

    prev_images = None
    prev_masks = None
    prev_audio = None

    final_images = torch.cat(final_frames, dim=0) if final_frames else None
    final_masks_tensor = torch.cat(final_masks, dim=0) if final_masks else None

    final_audio_dict = None
    if final_audio:
        sr = audio_sr

        shapes = [a.shape for a in final_audio]
        dims = [a.ndim for a in final_audio]
        if len(set(dims)) > 1:
            raise ValueError(
                f"Audio dimension mismatch: cannot combine arrays "
                f"with different shapes {shapes}")

        is_stereo = final_audio[0].ndim == 2 and final_audio[0].shape[0] == 2
        if is_stereo:
            audio_concat = np.concatenate(final_audio, axis=1)
            final_audio_dict = {"waveform": torch.from_numpy(
                audio_concat).unsqueeze(0).float(), "sample_rate": sr, }
        else:
            audio_concat = np.concatenate(final_audio, axis=0).reshape(1, -1)
            final_audio_dict = {
                "waveform": torch.from_numpy(audio_concat).float(),
                "sample_rate": sr,
            }

    output_path, frontend_data = av_save(
        images=final_images,
        masks=final_masks_tensor,
        audio=final_audio_dict,
        filename_prefix=filename_prefix,
        fps=fps,
        profile=profile)

    if not need_images:
        final_images = None
    if not need_masks:
        final_masks_tensor = None
    if not need_audio:
        final_audio_dict = None

    return (output_path, frontend_data, final_images, final_masks_tensor, final_audio_dict)
