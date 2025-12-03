import os
import folder_paths
from PIL import Image
import torch
import numpy as np
from comfy.utils import common_upscale
from .textOverlay import batch_draw_text
from functools import reduce
from .utils import mask_to_image, simple_blend

def frameIndexInfo(i, previous_count, chunk_index, chunk_count, total, overlap):
    chunk = chunk_index + 1
    is_overlap = True if chunk_index > 0 and i < overlap else False
    return (
        f"{str(previous_count + i + 1).zfill(len(str(total)))} / {total}", # frame_label
        f"{str(chunk).zfill(len(str(chunk_count)))} of {chunk_count}", # chunk_label
        is_overlap,
        f"chunks {chunk - 1} + {chunk}", # overlap_label
    )

def getOverlayConfigs(i, previous_count, chunk_index, chunk_count, total, w, h, length, overlap, fps):
    frame_label, chunk_label, is_overlap, overlap_label = frameIndexInfo(i, previous_count, chunk_index, chunk_count, total, overlap)
    configs = []
    configs.append(
        {
            "text": f"{frame_label}\n{chunk_label}",
            "vertical_alignment": "top",
            "horizontal_alignment": "right",
        },
    )
    configs.append(
        {
            "text": f"{w} x {h} @ {fps:.2f}FPS\nchunk_length: {length}\nchunk_overlap: {overlap}",
            "font_size": 12,
            "vertical_alignment": "bottom",
            "horizontal_alignment": "right",
        },
    )
    if is_overlap:
        configs.append(
            {
                "text": "OVERLAP",
                "font_size": 24,
                "fill_color_hex": "#FF0000",
                "stroke_color_hex": "#FFFFFF",
                "vertical_alignment": "top",
                "horizontal_alignment": "left",
            },
        )
        configs.append(
            {
                "text": overlap_label,
                "font_size": 14,
                "fill_color_hex": "#FF0000",
                "stroke_color_hex": "#FFFFFF",
                "vertical_alignment": "top",
                "horizontal_alignment": "left",
                "y_shift": 24 + 4,
            },
        )
    return configs

def overlay_debug(images, previous_count, chunk_index, chunk_count, chunk_length, chunk_overlap, total_length, fps):
    w = images.shape[2]
    h = images.shape[1]
    images = batch_draw_text(
        images,
        [getOverlayConfigs(i, previous_count, chunk_index, chunk_count, total_length, w, h, chunk_length, chunk_overlap, fps) for i in range(0, len(images))],
    )
    return images

def combine_images_and_masks(images, masks):
    masks = mask_to_image(masks) if masks is not None else None
    out = None
    if images is not None and masks is None: out = images
    if images is None and masks is not None: out = masks
    if images is not None and masks is not None: out = simple_blend(images, masks)
    #if images is not None and masks is not None: out = torch.cat((images, masks), dim=1) # side-by-side
    return out

def create_preview_video(images, masks, show_debug, d, c):
    previous_count = ((d["index"]) * (c["chunk_length"] - c["chunk_overlap"]))
    preview_video_chunk = combine_images_and_masks(images, masks)
    if show_debug:
        preview_video_chunk = overlay_debug(
            preview_video_chunk,
            previous_count,
            d["index"],
            c["chunk_count"],
            c["chunk_length"],
            c["chunk_overlap"],
            c["total_length"],
            d["fps"],
        )
    return preview_video_chunk
