import torch
import math
from ..lib.utils import count, log, force_wan_length, fix_total_length, get_this_chunk_length
from ..lib.utils_av import load, concat_audios
from ..lib.utils_tensor import monochrome_image, monochrome_mask, resize_image, resize_mask
from ..lib.utils_format import format_images, format_masks, format_audio, format_latents, format_fps
from ..lib.utils_performance import get_ts

class ChunkerDivide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["None", "Wan", "Wan-VACE"], {"tooltip": "Force chunk lengths to match Wan's format 4n+1. 16fps for Wan21, 24fps for Wan22"}),
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

    RETURN_TYPES = ("CHUNKER_DATA", "IMAGE", "MASK", "AUDIO", "LATENT", "INT", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("chunker_data", "images", "masks", "audio", "latents", "width", "height", "chunk_length", "chunk_overlap", "total_length", "chunk_count", "index")
    OUTPUT_TOOLTIPS = (
        "Connect \"chunker_data\" to the \"ChunkerCombine\" node",
        "Chunk of images",
        "Chunk of masks",
        "Chunk of audio",
        "Last chunk's latents",
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
    DESCRIPTION = "ChunkerDivide"

    @classmethod
    def IS_CHANGED(self, image):
        return float("NaN") # force run if cached, so start timestamp updates

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
        ts_chunk_start = get_ts()

        s = store if store is not None else {
            "index": 0,
            "last_chunk_path": None,
            "ts_chunk_starts": [],
            "last_latents": None,
        }

        out_fps = fps
        if out_fps is None and mode.startswith("Wan"): out_fps = 16.0
        if out_fps is None: out_fps = 30.0

        if total_length == 0:
            total_length = max(
                len(images) if images is not None else 0,
                len(masks) if masks is not None else 0,
            )

        if mode.startswith("Wan"):
            chunk_length = force_wan_length(chunk_length)
            total_length = fix_total_length(total_length, chunk_length, chunk_overlap)

        this_chunk_length = get_this_chunk_length(s["index"], chunk_length, chunk_overlap, total_length)

        w = None
        h = None

        start = (s["index"] * (chunk_length - chunk_overlap))
        end = start + chunk_length
        chunk_count = math.ceil((total_length - chunk_overlap) / (chunk_length - chunk_overlap))

        c = {
            "mode": mode,
            "chunk_length": chunk_length,
            "chunk_overlap": chunk_overlap,
            "total_length": total_length,
            "chunk_count": chunk_count,
        }

        log(f"Starting chunk {s["index"] + 1} of {c["chunk_count"]}...")

        out_images = []
        out_masks = []
        out_audio = []

        # get the overlap from the last chunk (video file) that Combine saved
        if s["last_chunk_path"] is not None and chunk_overlap > 0:
            overlap_images, overlap_masks, overlap_audio_dict, fps = load(
                path=s["last_chunk_path"],
                alpha_mode="2ndStream",
                start_n=-chunk_overlap,
            )
            w = overlap_images.shape[2]
            h = overlap_images.shape[1]
            if overlap_images is not None:
                out_images.append(overlap_images)
                if mode == "Wan-VACE":
                    # preserve overlap images with black masks
                    black_mask = monochrome_mask(w, h, 0)
                    out_masks.append(torch.cat([black_mask] * len(overlap_images)))
            if overlap_masks is not None:
                if mode != "Wan-VACE":
                    out_masks.append(overlap_masks)
            if overlap_audio_dict is not None: out_audio.append(overlap_audio_dict)


        # prepare chunk of images from input
        if images is not None:
            if w is None: w = images.shape[2]
            if h is None: h = images.shape[1]
            images_chunk = images[start + count(out_images):end]
            if (len(images_chunk) > 0): out_images.append(images_chunk)

        # prepare chunk of masks from input
        if masks is not None:
            out_masks.append(masks[start + count(out_masks):end])

        # prepare chunk of audio from input
        if audio is not None:
            samples_per_frame = math.floor(audio["sample_rate"] / fps)
            # astart = (start + count(out_images)) * samples_per_frame
            astart = start * samples_per_frame
            aend = end * samples_per_frame
            # print('end start',end,start)
            # print('aend astart',aend,astart)
            out_audio.append({
                "waveform": audio["waveform"][:,:,astart:aend],
                "sample_rate": audio["sample_rate"],
            })

        if w is None: w = 512
        if h is None: h = 512

        # for wan vace
        if mode == "Wan-VACE":
            # if more images than masks, add same amount of black masks to masks
            if count(out_images) > count(out_masks):
                black_mask = monochrome_mask(w, h, 0)
                out_masks.append(torch.cat([black_mask] * (count(out_images) - count(out_masks))))

            # if not enough images to fill chunk, add some grey images
            if count(out_images) < this_chunk_length:
                grey_image = monochrome_image(w, h, 0.5)
                out_images.append(torch.cat([grey_image] * (this_chunk_length - count(out_images))))

            # if not enough masks to fill chunk, add some white masks
            if count(out_masks) < this_chunk_length:
                white_mask = monochrome_mask(w, h, 1.0)
                out_masks.append(torch.cat([white_mask] * (this_chunk_length - count(out_masks))))

        # finalise out images, resize and concat together
        out_images_torch = None
        if len(out_images) > 0:
            out_images_resized = list(map(lambda tensor: resize_image(tensor, w, h), out_images))
            out_images_torch = torch.cat(out_images_resized)
            assert len(out_images_torch.shape) == 4, f"images are not rank 4 {out_images_torch.shape}, expected BHWC"

        # finalise out masks, resize and concat together
        out_masks_torch = None
        if len(out_masks) > 0:
            out_masks_resized = list(map(lambda tensor: resize_mask(tensor, w, h), out_masks))
            out_masks_torch = torch.cat(out_masks_resized)
            assert len(out_masks_torch.shape) == 3, f"masks are not rank 3 {out_masks_torch.shape}, expected BHW"

        # finalise out audio, concat together
        out_audio_dict = None
        if len(out_audio) > 0:
            out_audio_dict = concat_audios(out_audio)

        latents = s["last_latents"]

        chunker_data = {
            "start_node_id": unique_id,
            "index": s["index"],
            "chunker_config": c,
            "fps": out_fps,
            "ts_chunk_starts": [
                *s["ts_chunk_starts"],
                ts_chunk_start,
            ],
        }

        ui_values = {
            "input_label_values": {
                "images": format_images(images),
                "masks": format_masks(masks),
                "audio": format_audio(audio),
                "fps": format_fps(fps),
            },
            "output_label_values": {
                "images": format_images(out_images_torch),
                "masks": format_masks(out_masks_torch),
                "audio": format_audio(out_audio_dict),
                "latents": format_latents(latents),
                "width": w,
                "height": h,
                "chunk_length": this_chunk_length, # max(count(out_images), count(out_masks)),
                "chunk_overlap": chunk_overlap,
                "total_length": total_length,
                "chunk_count": chunk_count,
                "index": s["index"],
            },
        }

        return {
            "ui": {"values": [ui_values]},
            "result": (
                chunker_data,
                out_images_torch,
                out_masks_torch,
                out_audio_dict,
                latents,
                w,
                h,
                this_chunk_length, # max(count(out_images), count(out_masks)),
                chunk_overlap,
                total_length,
                chunk_count,
                s["index"],
            ),
        }
