from typing import Optional, Tuple
import torch
from .image_text_overlay import batch_draw_text
from .utils_tensor import mask_to_image


def frame_index_info(
    i: int,
    previous_count: int,
    chunk_index: int,
    chunk_count: int,
    chunk_length: int,
    total: int,
    overlap: int,
) -> Tuple[str, str, bool]:
    chunk = chunk_index + 1
    is_first_chunk = chunk_index == 0
    is_last_chunk = chunk_index == chunk_count - 1
    is_overlap = (
        not is_last_chunk and i >= chunk_length -
        overlap) or (
        not is_first_chunk and i < overlap)
    return (
        f"{str(previous_count + i + 1).zfill(len(str(total)))} / {total}",
        f"{str(chunk).zfill(len(str(chunk_count)))} of {chunk_count}",
        is_overlap,
    )


def get_overlay_config(
    i: int,
    previous_count: int,
    chunk_index: int,
    chunk_count: int,
    total: int,
    w: int,
    h: int,
    chunk_length: int,
    overlap: int,
    fps: float,
    overlap_blend_mode: str,
    audio_layout: str,
) -> list[dict]:
    frame_label, chunk_label, is_overlap = frame_index_info(
        i, previous_count, chunk_index, chunk_count,
        chunk_length, total, overlap,
    )
    configs = []
    em = h / 512
    configs.append(
        {
            "text": f"{frame_label}\n{chunk_label}",
            "font_size": int(em * 20),
            "vertical_alignment": "top",
            "horizontal_alignment": "right",
        },
    )
    configs.append(
        {
            "text": (
                f"{w} x {h} @ {fps:.2f}FPS\n{audio_layout}\n"
                f"chunk_length: {chunk_length}\nchunk_overlap: {overlap}\n"
                f"overlap_blend_mode: {overlap_blend_mode}"
            ),
            "font_size": int(
                em *
                16),
            "vertical_alignment": "bottom",
            "horizontal_alignment": "right",
        },
    )
    if is_overlap:
        configs.append(
            {
                "text": "OVERLAP",
                "font_size": int(em * 28),
                "fill_color_hex": "#FF0000",
                "stroke_color_hex": "#FFFFFF",
                "vertical_alignment": "top",
                "horizontal_alignment": "left",
            },
        )
    return configs


def overlay_debug_text(
    images: torch.Tensor,
    previous_count: int,
    chunk_index: int,
    chunk_count: int,
    chunk_length: int,
    chunk_overlap: int,
    total_length: int,
    fps: float,
    overlap_blend_mode: str,
    audio_layout: str,
) -> torch.Tensor:
    w = images.shape[2]
    h = images.shape[1]
    config = [
        get_overlay_config(
            i,
            previous_count,
            chunk_index,
            chunk_count,
            total_length,
            w,
            h,
            chunk_length,
            chunk_overlap,
            fps,
            overlap_blend_mode,
            audio_layout) for i in range(
            0,
            len(images))]
    images = batch_draw_text(images, config)
    return images


def combine_images_and_masks(
        images: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if images is not None and masks is not None:
        alpha = masks.unsqueeze(-1)
        return torch.cat([images, alpha], dim=-1)
    if images is not None:
        alpha = torch.ones(
            images.shape[0],
            images.shape[1],
            images.shape[2],
            1,
            dtype=images.dtype)
        return torch.cat([images, alpha], dim=-1)
    if masks is not None:
        rgb = mask_to_image(masks)
        alpha = masks.unsqueeze(-1)
        return torch.cat([rgb, alpha], dim=-1)
    return None


def create_preview_video(
    images: Optional[torch.Tensor],
    masks: Optional[torch.Tensor],
    audio: Optional[dict],
    d: dict,
    c: dict,
    overlap_blend_mode: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict], int]:
    previous_count = ((d["index"]) * (c["chunk_length"] - c["chunk_overlap"]))
    preview_video_chunk = combine_images_and_masks(images, masks)
    audio_channel_count = (
        audio["waveform"].shape[1] if audio is not None else 0
    )
    audio_layout = ["", "mono", "stereo"][audio_channel_count]
    if audio is not None:
        audio_layout = f"{audio['sample_rate']}Hz {audio_layout}"
    preview_video_chunk = overlay_debug_text(
        preview_video_chunk,
        previous_count,
        d["index"],
        c["chunk_count"],
        c["chunk_length"],
        c["chunk_overlap"],
        c["total_length"],
        d["fps"],
        overlap_blend_mode,
        audio_layout,
    )
    preview_masks = preview_video_chunk[:, :, :, 3] if preview_video_chunk.shape[3] == 4 else None
    return (preview_video_chunk, preview_masks, audio, d["fps"])
