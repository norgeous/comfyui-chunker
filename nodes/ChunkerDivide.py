import torch
import math
from ..lib.utils import (
    count,
    log,
    resize_image,
    resize_mask,
    force_wan_length,
    fix_total_length,
    get_this_chunk_length,
)
from ..lib.av.load_audio import concat_audios
from ..lib.utils_format import format_audio, format_fps
from ..lib.utils_av import load

class ChunkerDivide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["None", "Wan21", "Wan22"], {"tooltip": "Force chunk lengths to match Wan's format 4n+1. 16fps for Wan21, 24fps for Wan22"}),
                "chunk_length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 1, "tooltip": "Count of images in each chunk"}),
                "chunk_overlap": ("INT", {"default": 4, "min": 0, "max": 4096, "step": 1, "tooltip": "Count of images to overlap between chunks"}),
                "total_length": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "Minimum count of images in the final output. 0 to use the images length"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "images"}),
                "masks": ("MASK", {"tooltip": "masks"}),
                "audio": ("AUDIO", {"tooltip": "audio"}),
                "fps": ("FLOAT", {"forceInput": True, "tooltip": "fps"}),
                "store": ("*",), # hidden by js
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("CHUNKER_DATA", "IMAGE", "MASK", "AUDIO", "INT", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("chunker_data", "images", "masks", "audio", "width", "height", "chunk_length", "chunk_overlap", "total_length", "chunk_count", "index")
    OUTPUT_TOOLTIPS = (
        "Connect \"chunker_data\" to the \"ChunkerCombine\" node",
        "Chunk of images",
        "Chunk of masks",
        "Chunk of audio",
        "Width of images",
        "Height of images",
        "Count of images in each chunk",
        "Count of images to overlap between each chunk",
        "Total length of output images",
        "Count of chunks",
        "The current itteration index, ie; 0, 1, 2, ...",
    )
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    DESCRIPTION = "Chunker"

    def execute(
        self,
        mode,
        chunk_length,
        chunk_overlap,
        total_length,
        images=None,
        masks=None,
        audio=None,
        fps=None,
        store=None,
        unique_id=None,
    ):
        s = store if store is not None else {
            "index": 0,
            "last_chunk_path": None,
        }

        if fps is None and mode == "Wan21": fps = 16
        if fps is None and mode == "Wan22": fps = 24
        if fps is None: fps = 30

        if total_length == 0:
            total_length = max(
                len(images) if images is not None else 0,
                len(masks) if images is not None else 0,
            )

        if mode == "Wan21" or mode == "Wan22":
            chunk_length = force_wan_length(chunk_length)
            total_length = fix_total_length(total_length, chunk_length, chunk_overlap)

        this_chunk_length = get_this_chunk_length(s["index"], chunk_length, chunk_overlap, total_length)

        w = None
        h = None

        start = (s["index"] * (chunk_length - chunk_overlap))
        end = start + chunk_length
        chunk_count = math.ceil((total_length - chunk_overlap) / (chunk_length - chunk_overlap))

        out_images = []
        out_masks = []
        out_audio = []

        # get the overlap from the last chunk that Combine saved
        if s["last_chunk_path"] is not None:
            overlap_images, overlap_masks, overlap_audio_dict, fps = load(path=s["last_chunk_path"], start_n=-chunk_overlap)
            w = overlap_images.shape[2]
            h = overlap_images.shape[1]
            out_images.append(overlap_images)
            out_masks.append(overlap_masks)
            out_audio.append(overlap_audio_dict)

        if (mode == "Wan21" or mode == "Wan22") and (count(out_images) > count(out_masks)):
            black_panel = torch.full((1, h, w), 0) # panel_mask(w, h, 0)
            out_masks.append(torch.cat([black_panel] * (count(out_images) - count(out_masks)))) # add same amount of black masks to masks

        if audio is not None:
            samples_per_frame = math.floor(audio["sample_rate"] / fps)
            astart = (start + count(out_images)) * samples_per_frame
            aend = end * samples_per_frame
            out_audio.append({
                "waveform": audio["waveform"][:,:,astart:aend],
                "sample_rate": audio["sample_rate"],
            })

        if images is not None:
            if w is None: w = images.shape[2]
            if h is None: h = images.shape[1]
            out_images.append(images[start + count(out_images):end])

        if masks is not None:
            out_masks.append(masks[start + count(out_masks):end])

        if w is None: w = 512
        if h is None: h = 512

        grey_panel = torch.full((1, h, w, 3), 0.5) # panel_image(w, h, 127, 127, 127)
        white_panel = torch.full((1, h, w), 1) # panel_mask(w, h, 255)
        black_panel = torch.full((1, h, w), 0) # panel_mask(w, h, 0)

        # do some stuff for Wan
        if mode == "Wan21" or mode == "Wan22":
            # if not enough images, invent some blank (grey) ones (for t2v)
            if count(out_images) < this_chunk_length: out_images.append(torch.cat([grey_panel] * (this_chunk_length - count(out_images))))

            # if not enough masks, invent some blank (white) ones (for t2v)
            if count(out_masks) < this_chunk_length: out_masks.append(torch.cat([white_panel] * (this_chunk_length - count(out_masks))))

        out_images_torch = None
        if len(out_images) > 0:
            out_images_resized = list(map(lambda tensor: resize_image(tensor, w, h), out_images))
            out_images_torch = torch.cat(out_images_resized)
            assert len(out_images_torch.shape) == 4, f"images are not rank 4 {out_images_torch.shape}"

        out_masks_torch = None
        if len(out_masks) > 0:
            out_masks_resized = list(map(lambda tensor: resize_mask(tensor, w, h), out_masks))
            out_masks_torch = torch.cat(out_masks_resized)
            assert len(out_masks_torch.shape) == 3, f"masks are not rank 3 {out_masks_torch.shape}"

        out_audio_dict = None
        if len(out_audio) > 0:
            out_audio_dict = concat_audios(out_audio)

        c = {
            "mode": mode,
            "chunk_length": chunk_length,
            "chunk_overlap": chunk_overlap,
            "total_length": total_length,
            "chunk_count": chunk_count,
        }

        chunker_data = {
            "start_node_id": unique_id,
            "index": s["index"],
            "chunker_config": c,
            "fps": fps,
        }

        ui_values = {
            "input_label_values": {
                "images": len(images) if images is not None else 0,
                "masks": len(masks) if masks is not None else 0,
                "audio": format_audio(audio),
                "fps": format_fps(fps),
            },
            "output_label_values": {
                "images": count(out_images),
                "masks": count(out_masks),
                "audio": format_audio(out_audio_dict),
                "width": w,
                "height": h,
                "chunk_length": max(count(out_images), count(out_masks)),
                "chunk_overlap": chunk_overlap,
                "total_length": total_length,
                "chunk_count": chunk_count,
                "index": s["index"],
            },
        }

        log(f"Starting chunk {s["index"] + 1} of {c["chunk_count"]}...")

        return {
            "ui": {"values": [ui_values]},
            "result": (
                chunker_data,
                out_images_torch,
                out_masks_torch,
                out_audio_dict,
                w,
                h,
                max(count(out_images), count(out_masks)),
                chunk_overlap,
                total_length,
                chunk_count,
                s["index"],
            ),
        }
