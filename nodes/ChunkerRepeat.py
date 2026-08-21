import torch
import math
from comfy_api.latest import io
from ..lib.utils import count, log
from ..lib.plan_chunks import plan_chunks
from ..lib.av_load import av_load
from ..lib.utils_comfy import concat_audios
from ..lib.utils_tensor import resize_image, resize_mask
from ..lib.utils_format import (
    format_images, format_masks, format_audio, format_fps)
from ..lib.utils_performance import get_ts
from enum import Enum


class Mode(Enum):
    DEFAULT = "default"
    WAN2 = "wan2"
    LTX2 = "ltx2"
    MINIMAX_H3 = "minimax-h3"


mode_settings = {
    Mode.DEFAULT: {
        "dimension_adjuster": lambda length: (length // 2) * 2, # 2n
        "length_adjuster": lambda length: length, # n
        "fps": 30.0,
    },
    Mode.WAN2: {
        "dimension_adjuster": lambda length: (length // 16) * 16, # 16n
        "length_adjuster": lambda length: (math.ceil((length - 1) / 4) * 4) + 1, # 4n+1. example: 1, 5, 9, 13, 17
        "fps": 16.0,
    },
    Mode.LTX2: {
        "dimension_adjuster": lambda length: (length // 32) * 32, # 32n
        "length_adjuster": lambda length: (math.ceil((length - 1) / 8) * 8) + 1, # 8n+1. example: 1, 9, 17, 25, 33
        "fps": 25.0,
    },
    Mode.MINIMAX_H3: {
        "dimension_adjuster": lambda length: (length // 32) * 32, # 32n
        "length_adjuster": lambda length: (math.ceil((length - 5) / 17) * 17) + 5, # 17n+5. example: 5, 22, 39, 56, 73
        "fps": 24.0,
    },
}


class ChunkerRepeat(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ChunkerRepeat",
            display_name="\U0001F36B Repeat",
            category="chunker",
            description=(
                "Repeat nodes between this node and `🍫 Combine`. "
                "Optionally divide long batches images, masks and / or audio into smaller chunks "
                "and process the chunks sequentially. Optionally use the end of last chunk "
                "as start of this chunk (with `chunk_overlap`)."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    optional=True,
                    tooltip="images (optional)",
                ),
                io.Mask.Input(
                    "masks",
                    optional=True,
                    tooltip="masks (optional)",
                ),
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="audio (optional)",
                ),
                io.Float.Input(
                    "fps",
                    optional=True,
                    force_input=True,
                    tooltip="The default FPS of 30 is overridden when `mode` is not default (see mode tooltip). If you supply a value it overrides the FPS value from mode",
                ),
                io.Combo.Input(
                    "mode",
                    options=list(map(lambda member: member.value, Mode)),
                    tooltip=(
                        "Adjust chunk_length, total_length, image dimensions and default FPS to match selected format\n"
                        "\n"
                        "default:          dim 2n,   length n,     fps 30\n"
                        "wan2:            dim 16n,  length 4n+1,  fps 16\n"
                        "ltx2:               dim 32n,  length 8n+1,  fps 25\n"
                        "minimax-h3: dim 32n,  length 17n+5, fps 24"
                    ),
                ),
                io.Int.Input(
                    "chunk_length",
                    tooltip="Count of images in each chunk",
                    default=81,
                    min=1,
                    max=4096,
                    step=1,
                ),
                io.Int.Input(
                    "chunk_overlap",
                    tooltip="Count of images to overlap between chunks",
                    default=4,
                    min=0,
                    max=4096,
                    step=1,
                ),
                io.DynamicCombo.Input(
                    "repeat_until",
                    tooltip="How to determine total output length",
                    options=[
                        io.DynamicCombo.Option("chunk_count", [
                            io.Int.Input(
                                "chunk_count",
                                tooltip="Number of chunks in the final output",
                                default=2,
                                min=1,
                                max=1000,
                                step=1,
                            ),
                        ]),
                        io.DynamicCombo.Option("total_length", [
                            io.Int.Input(
                                "total_length",
                                tooltip="Minimum count of images in the final output",
                                default=100,
                                min=1,
                                max=10000,
                                step=1,
                            ),
                        ]),
                        io.DynamicCombo.Option("input_length", []),
                    ],
                ),
                io.Custom("*").Input(
                    "store",
                    optional=True,
                ),
            ],
            outputs=[
                io.Custom("CHUNKER_DATA").Output(
                    "chunker_data",
                    tooltip=("Connect \"chunker_data\" to the \"ChunkerCombine\" node"),
                ),
                io.Image.Output(
                    "images",
                    tooltip="Chunk of images",
                ),
                io.Mask.Output(
                    "masks",
                    tooltip="Chunk of masks",
                ),
                io.Audio.Output(
                    "audio",
                    tooltip="Chunk of audio",
                ),

            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        self,
        mode,
        chunk_length,
        chunk_overlap,
        repeat_until,
        images=None,
        masks=None,
        audio=None,
        fps=None,
        store=None,
    ) -> io.NodeOutput:
        ts_chunk_start = get_ts()

        s = store if store is not None else {
            "index": 0,
            "last_chunk_path": None,
            "ts_chunk_starts": [],
        }

        settings = mode_settings[Mode(mode)]

        out_fps = fps
        if out_fps is None:
            out_fps = settings["fps"]

        # resolve total_length from repeat_until
        tl_type = repeat_until["repeat_until"]
        if tl_type == "chunk_count":
            target_count = repeat_until["chunk_count"]
            adjusted = settings["length_adjuster"](chunk_length)
            total_length = (target_count - 1) * (adjusted - chunk_overlap) + adjusted
        elif tl_type == "input_length":
            total_length = max(
                len(images) if images is not None else 0,
                len(masks) if masks is not None else 0,
                audio["waveform"].shape[-1] // math.floor(audio["sample_rate"] / out_fps) if audio is not None else 0,
            )
        else:  # total_length
            total_length = repeat_until["total_length"]

        chunk_length, total_length, chunk_lengths = plan_chunks(
            settings["length_adjuster"],
            chunk_length,
            chunk_overlap,
            total_length,
        )
        this_chunk_length = chunk_lengths[s["index"]]

        w = None
        h = None

        start = (s["index"] * (chunk_length - chunk_overlap))
        end = start + chunk_length
        chunk_count = math.ceil(
            (total_length - chunk_overlap) / (chunk_length - chunk_overlap))

        c = {
            "mode": mode,
            "chunk_length": chunk_length,
            "chunk_overlap": chunk_overlap,
            "total_length": total_length,
            "chunk_count": chunk_count,
        }

        log(f"ChunkerRepeat#{self.hidden.unique_id}: Starting chunk {s['index'] + 1} of {c['chunk_count']}...")

        out_images = []
        out_masks = []
        out_audio = []

        # get the overlap from the last chunk (video file) that Combine saved
        if s["last_chunk_path"] is not None and chunk_overlap > 0:
            overlap_images, overlap_masks, overlap_audio_dict, _ = av_load(
                path=s["last_chunk_path"],
                start=-chunk_overlap,
            )
            w = overlap_images.shape[2]
            h = overlap_images.shape[1]
            if overlap_images is not None:
                out_images.append(overlap_images)
            if overlap_masks is not None:
                out_masks.append(overlap_masks)
            if overlap_audio_dict is not None:
                out_audio.append(overlap_audio_dict)

        # prepare chunk of images from input
        if images is not None:
            if w is None:
                w = settings["dimension_adjuster"](images.shape[2])
            if h is None:
                h = settings["dimension_adjuster"](images.shape[1])
            images_chunk = images[start + count(out_images):end]
            if (len(images_chunk) > 0):
                out_images.append(images_chunk)

        # prepare chunk of masks from input
        if masks is not None:
            out_masks.append(masks[start + count(out_masks):end])

        # prepare chunk of audio from input
        if audio is not None:
            samples_per_frame = math.floor(audio["sample_rate"] / out_fps)
            samples_already_collected = (
                out_audio[0]["waveform"].shape[-1]
                if len(out_audio) > 0 else 0
            )
            astart = (start * samples_per_frame) + samples_already_collected
            aend = end * samples_per_frame
            out_audio.append({
                "waveform": audio["waveform"][:, :, astart:aend],
                "sample_rate": audio["sample_rate"],
            })

        if w is None:
            w = 512
        if h is None:
            h = 512

        # finalise out images, resize and concat together
        out_images_torch = None
        if len(out_images) > 0:
            out_images_resized = list(
                map(lambda tensor: resize_image(tensor, w, h), out_images))
            out_images_torch = torch.cat(out_images_resized)

        # finalise out masks, resize and concat together
        out_masks_torch = None
        if len(out_masks) > 0:
            out_masks_resized = list(
                map(lambda tensor: resize_mask(tensor, w, h), out_masks))
            out_masks_torch = torch.cat(out_masks_resized)

        # finalise out audio, concat together
        out_audio_dict = None
        if len(out_audio) > 0:
            out_audio_dict = concat_audios(out_audio)

        chunker_data = {
            "start_node_id": self.hidden.unique_id,
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
            },
        }

        return io.NodeOutput(
            chunker_data,
            out_images_torch,
            out_masks_torch,
            out_audio_dict,
            ui={"values": [ui_values]},
        )
