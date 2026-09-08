import torch
import math
import safetensors.torch
from comfy_api.latest import io
from ..lib.utils import count, log
from ..lib.plan_chunks import plan_chunks
from ..lib.av_load import av_load
from ..lib.utils_comfy import concat_audios
from ..lib.utils_tensor import resize_image, resize_mask
from ..lib.utils_format import (format_images, format_masks, format_audio, format_fps, format_video, format_latent)
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
        "chunk_length": {"default": 100, "min": 1, "step": 1},   # n
        "audio_latents_per_second": 0,
        "length_to_latent_length": lambda pixel_length: (pixel_length + 3) // 4,
        "pixel_to_latent_range": lambda pixel_start, pixel_end, total_pixels, total_latents: (pixel_start * total_latents // total_pixels, pixel_end * total_latents // total_pixels),
    },
    Mode.WAN2: {
        "dimension_adjuster": lambda length: (length // 16) * 16, # 16n
        "length_adjuster": lambda length: (math.ceil((length - 1) / 4) * 4) + 1, # 4n+1. example: 1, 5, 9, 13, 17
        "fps": 16.0,
        "chunk_length": {"default": 81, "min": 1, "step": 4},   # 4n+1
        "audio_latents_per_second": 0,
        "length_to_latent_length": lambda pixel_length: (pixel_length + 3) // 4,
        "pixel_to_latent_range": lambda pixel_start, pixel_end, total_pixels, total_latents: (pixel_start * total_latents // total_pixels, pixel_end * total_latents // total_pixels),
    },
    Mode.LTX2: {
        "dimension_adjuster": lambda length: (length // 32) * 32, # 32n
        "length_adjuster": lambda length: (math.ceil((length - 1) / 8) * 8) + 1, # 8n+1. example: 1, 9, 17, 25, 33
        "fps": 25.0,
        "chunk_length": {"default": 81, "min": 1, "step": 8},   # 8n+1
        "audio_latents_per_second": 25,
        "length_to_latent_length": lambda pixel_length: (pixel_length + 7) // 8,
        "pixel_to_latent_range": lambda pixel_start, pixel_end, total_pixels, total_latents: (pixel_start * total_latents // total_pixels, pixel_end * total_latents // total_pixels),
    },
    Mode.MINIMAX_H3: {
        "dimension_adjuster": lambda length: (length // 32) * 32, # 32n
        "length_adjuster": lambda length: (math.ceil((length - 5) / 17) * 17) + 5, # 17n+5. example: 5, 22, 39, 56, 73
        "fps": 24.0,
        "chunk_length": {"default": 107, "min": 5, "step": 17},  # 17n+5
        "audio_latents_per_second": 40,
        "length_to_latent_length": lambda pixel_length: max(0, (math.ceil(pixel_length / 17) * 5) - 3),
        "pixel_to_latent_range": lambda pixel_start, pixel_end, total_pixels, total_latents: (
            lambda clip_start, clip_end, total_clips: (
                clip_start * 5,
                min((clip_end + 1) * 5 - (3 if clip_end == total_clips - 1 else 0), total_latents)
            )
        )(pixel_start // 17, (pixel_end - 1) // 17, (total_pixels + 16) // 17),
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
                "as start of this chunk (with `overlap_length`)."
            ),
            inputs=[
                io.Video.Input(
                    "video",
                    optional=True,
                    tooltip="video (optional)",
                ),
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
                io.Latent.Input(
                    "latent",
                    optional=True,
                    tooltip="Input latent chunk (optional)",
                ),
                io.Float.Input(
                    "fps",
                    optional=True,
                    force_input=True,
                    tooltip="The default FPS of 30 is overridden when `mode` is not default (see mode tooltip). If you supply a value it overrides the FPS value from mode",
                ),
                io.DynamicCombo.Input(
                    "mode",
                    options=[
                        io.DynamicCombo.Option(
                            member.value,
                            [
                                io.Int.Input(
                                    "chunk_length",
                                    tooltip="Count of images in each chunk",
                                    **mode_settings[member]["chunk_length"],
                                ),
                            ],
                        )
                        for member in Mode
                    ],
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
                    "overlap_length",
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
                io.Latent.Output(
                    "latent",
                    tooltip="Latent from previous chunk",
                ),

            ],
            hidden=[io.Hidden.unique_id, io.Hidden.dynprompt],
        )

    @classmethod
    def execute(
        self,
        mode,
        overlap_length,
        repeat_until,
        video=None,
        images=None,
        masks=None,
        audio=None,
        latent=None,
        fps=None,
        store=None,
    ) -> io.NodeOutput:
        ts_chunk_start = get_ts()

        selected_mode = Mode(mode["mode"])
        settings = mode_settings[selected_mode]
        chunk_length = mode["chunk_length"]

        s = store if store is not None else {
            "index": 0,
            "last_chunk_path": None,
            "ts_chunk_starts": [],
        }

        video_source = None
        video_fps = None
        video_frame_count = None
        if video is not None:
            video_source = video.get_stream_source()
            video_fps = float(video.get_frame_rate())
            video_frame_count = video.get_frame_count()

        out_fps = fps
        if out_fps is None:
            out_fps = video_fps if video_fps is not None else settings["fps"]

        # resolve total_length from repeat_until
        tl_type = repeat_until["repeat_until"]
        if tl_type == "chunk_count":
            target_count = repeat_until["chunk_count"]
            adjusted = settings["length_adjuster"](chunk_length)
            total_length = (target_count - 1) * (adjusted - overlap_length) + adjusted
        elif tl_type == "input_length":
            total_length = max(
                len(images) if images is not None else 0,
                len(masks) if masks is not None else 0,
                video_frame_count if video_frame_count is not None else 0,
                audio["waveform"].shape[-1] // math.floor(audio["sample_rate"] / out_fps) if audio is not None else 0,
            )
        else:  # total_length
            total_length = repeat_until["total_length"]

        chunk_length, total_length, chunk_lengths = plan_chunks(
            settings["length_adjuster"],
            chunk_length,
            overlap_length,
            total_length,
        )
        this_chunk_length = chunk_lengths[s["index"]]

        w = None
        h = None

        start = (s["index"] * (chunk_length - overlap_length))
        end = start + chunk_length
        chunk_count = math.ceil(
            (total_length - overlap_length) / (chunk_length - overlap_length))

        c = {
            "mode": selected_mode.value,
            "chunk_length": chunk_length,
            "overlap_length": overlap_length,
            "total_length": total_length,
            "chunk_count": chunk_count,
        }

        log(f"ChunkerRepeat#{self.hidden.dynprompt.get_display_node_id(self.hidden.unique_id)}: Starting chunk {s['index'] + 1} of {c['chunk_count']}...")

        # load latent overlap from previous chunk's safetensors
        overlap_latent = None
        overlap_latent_count = 0
        audio_overlap_start = 0
        audio_overlap_end = 0
        video_overlap_count = 0
        audio_overlap_count = 0
        if s.get("last_latent_path") is not None and overlap_length > 0:
            with safetensors.safe_open(s["last_latent_path"], framework="pt") as f:
                latent_type = f.metadata().get("type", "standard")
                
                latent_keys = [k for k in f.keys() if k.startswith("latent_")]
                if latent_keys:
                    # NestedTensor format (video + audio)
                    latent_keys.sort(key=lambda x: int(x.split("_")[1]))
                    tensors = [f.get_tensor(k) for k in latent_keys]
                    from comfy.nested_tensor import NestedTensor
                    full_latent_tensor = NestedTensor(tensors)
                else:
                    full_latent_tensor = f.get_tensor("latent")
                
                if full_latent_tensor is not None:
                    # Get expected total latent frames for configured total_length
                    expected_total_latents = settings["length_to_latent_length"](c["total_length"])
                    
                    # Use accurate pixel-to-latent mapping for overlap count
                    pixel_to_latent = settings["pixel_to_latent_range"]
                    overlap_pixel_start = max(0, start - overlap_length)
                    overlap_pixel_end = start
                    video_overlap_start, video_overlap_end = pixel_to_latent(
                        overlap_pixel_start, overlap_pixel_end, c["total_length"], expected_total_latents
                    )
                    audio_latents_per_sec = settings["audio_latents_per_second"]
                    audio_overlap_start = round(overlap_pixel_start / out_fps * audio_latents_per_sec) if audio_latents_per_sec > 0 else 0
                    audio_overlap_end = round(overlap_pixel_end / out_fps * audio_latents_per_sec) if audio_latents_per_sec > 0 else 0
                    
                    # H3 VAE uses token_overlap=2 for 5-frame overlap
                    token_overlap = settings.get("token_overlap", 2)
                    if video_overlap_end - video_overlap_start > token_overlap:
                        video_overlap_start = video_overlap_end - token_overlap

                    video_overlap_count = video_overlap_end - video_overlap_start
                    audio_overlap_count = audio_overlap_end - audio_overlap_start

                    log(f"ChunkerRepeat#{self.hidden.dynprompt.get_display_node_id(self.hidden.unique_id)}: Latent overlap indexes: overlap_pixel=[{overlap_pixel_start}, {overlap_pixel_end}], video_latent_count={video_overlap_count}, audio_latent_count={audio_overlap_count}")

                    # Slice overlap from the end of previous chunk's latent
                    if hasattr(full_latent_tensor, "tensors"):  # NestedTensor
                        video_t = full_latent_tensor.tensors[0]  # [B, 24, T, H, W]
                        audio_t = full_latent_tensor.tensors[1]  # [B, 32, 2, T]
                        
                        file_t = video_t.shape[2]
                        video_overlap_start = file_t - video_overlap_count
                        video_overlap_end = file_t
                        video_overlap = video_t[:, :, video_overlap_start:video_overlap_end, :, :]
                        if audio_latents_per_sec > 0:
                            audio_file_t = audio_t.shape[3]
                            audio_overlap_start = audio_file_t - audio_overlap_count
                            audio_overlap_end = audio_file_t
                        audio_overlap = audio_t[:, :, :, audio_overlap_start:audio_overlap_end] if audio_latents_per_sec > 0 else None
                        
                        overlap_tensors = []
                        if video_overlap.shape[2] > 0:
                            overlap_tensors.append(video_overlap)
                        if audio_overlap is not None and audio_overlap.shape[3] > 0:
                            overlap_tensors.append(audio_overlap)
                        
                        if overlap_tensors:
                            overlap_latent_tensor = NestedTensor(overlap_tensors)
                        else:
                            overlap_latent_tensor = None
                    else:
                        # Regular tensor (video only)
                        file_t = full_latent_tensor.shape[2]
                        video_overlap_start = file_t - video_overlap_count
                        video_overlap_end = file_t
                        overlap_latent_tensor = full_latent_tensor[:, :, video_overlap_start:video_overlap_end, :, :]
                    
                    if overlap_latent_tensor is not None:
                        overlap_latent = {"samples": overlap_latent_tensor}
                        if latent_type != "standard":
                            overlap_latent["type"] = latent_type

        out_images = []
        out_masks = []
        out_audio = []

        # get the overlap from the last chunk (video file) that Combine saved
        if s["last_chunk_path"] is not None and overlap_length > 0:
            overlap_images, overlap_masks, overlap_audio_dict, _ = av_load(
                path=s["last_chunk_path"],
                start=-overlap_length,
            )
            w = overlap_images.shape[2]
            h = overlap_images.shape[1]
            if overlap_images is not None:
                out_images.append(overlap_images)
            if overlap_masks is not None:
                out_masks.append(overlap_masks)
            if overlap_audio_dict is not None:
                out_audio.append(overlap_audio_dict)

        # load chunk frames from video (lazy load via av_load)
        if video_source is not None and images is None:
            load_start = start + count(out_images)
            load_end = end
            if load_start < load_end:
                video_images, video_masks, video_audio_dict, loaded_fps = av_load(
                    path=video_source,
                    start=load_start,
                    end=load_end,
                )
                if video_images is not None:
                    if w is None:
                        w = settings["dimension_adjuster"](video_images.shape[2])
                    if h is None:
                        h = settings["dimension_adjuster"](video_images.shape[1])
                    out_images.append(video_images)
                if video_masks is not None:
                    out_masks.append(video_masks)
                if video_audio_dict is not None:
                    out_audio.append(video_audio_dict)
                if fps is None and loaded_fps:
                    out_fps = loaded_fps

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

        # prepare chunk of latent from input
        input_latent_chunk = None
        if latent is not None:
            latent_dict = latent if isinstance(latent, dict) else {"samples": latent}
            full_input_latent = latent_dict["samples"]
            
            # Get expected total latent frames for configured total_length
            expected_total_latents = settings["length_to_latent_length"](c["total_length"])
            
            # Use accurate pixel-to-latent mapping from mode settings
            pixel_to_latent = settings["pixel_to_latent_range"]
            video_latent_start, video_latent_end = pixel_to_latent(start, end, c["total_length"], expected_total_latents)
            audio_latents_per_sec = settings["audio_latents_per_second"]
            audio_latent_start = round(start / out_fps * audio_latents_per_sec) if audio_latents_per_sec > 0 else 0
            audio_latent_end = round(end / out_fps * audio_latents_per_sec) if audio_latents_per_sec > 0 else 0
            
            # Apply boundary token drop for partial clips (H3 VAE token_drop at clip boundaries)
            boundary_drop = 0
            if c["mode"] == "minimax-h3" and end < c["total_length"]:
                clip_length = 17
                vae_ratio_t = 4
                frames_in_clip = end - ((end - 1) // clip_length) * clip_length
                if frames_in_clip != clip_length:
                    boundary_drop = math.ceil((clip_length - frames_in_clip) / vae_ratio_t)
            
            video_latent_end -= boundary_drop

            log(f"ChunkerRepeat#{self.hidden.dynprompt.get_display_node_id(self.hidden.unique_id)}: Chunk latent indexes: video_latent=[{video_latent_start}, {video_latent_end}], audio_latent=[{audio_latent_start}, {audio_latent_end}], boundary_drop={boundary_drop}")

            audio_latents_per_sec = settings["audio_latents_per_second"]
            audio_latent_start = round(start / out_fps * audio_latents_per_sec) if audio_latents_per_sec > 0 else 0
            audio_latent_end = round(end / out_fps * audio_latents_per_sec) if audio_latents_per_sec > 0 else 0
            
            # Calculate overlap latent frame count for trimming main chunk
            if overlap_latent is not None:
                if hasattr(overlap_latent["samples"], "tensors"):  # NestedTensor
                    overlap_latent_count = overlap_latent["samples"].tensors[0].shape[2]
                else:
                    overlap_latent_count = overlap_latent["samples"].shape[2]
            else:
                overlap_latent_count = 0

            log(f"ChunkerRepeat#{self.hidden.dynprompt.get_display_node_id(self.hidden.unique_id)}: overlap_latent_count={overlap_latent_count}")

            if hasattr(full_input_latent, "tensors"):  # NestedTensor
                video_t = full_input_latent.tensors[0]  # [B, 24, T, H, W]
                audio_t = full_input_latent.tensors[1]  # [B, 32, 2, T]
                
                video_chunk = video_t[:, :, video_latent_start + overlap_latent_count:video_latent_end, :, :]
                audio_chunk = audio_t[:, :, :, audio_latent_start + overlap_latent_count:audio_latent_end] if audio_latents_per_sec > 0 else None
                
                chunk_tensors = []
                if video_chunk.shape[2] > 0:
                    chunk_tensors.append(video_chunk)
                if audio_chunk is not None and audio_chunk.shape[3] > 0:
                    chunk_tensors.append(audio_chunk)
                
                if chunk_tensors:
                    from comfy.nested_tensor import NestedTensor
                    input_latent_chunk = NestedTensor(chunk_tensors)
                else:
                    input_latent_chunk = None
            else:
                # Regular tensor (video only)
                input_latent_chunk = full_input_latent[:, :, video_latent_start + overlap_latent_count:video_latent_end, :, :]

        if w is None:
            w = 512
        if h is None:
            h = 512

        # Combine overlap_latent + input_latent_chunk
        output_latent = None
        if overlap_latent is not None and input_latent_chunk is not None:
            # Prepend overlap to input chunk
            if hasattr(overlap_latent["samples"], "tensors") and hasattr(input_latent_chunk, "tensors"):
                # Both NestedTensor
                overlap_video = overlap_latent["samples"].tensors[0]
                overlap_audio = overlap_latent["samples"].tensors[1] if len(overlap_latent["samples"].tensors) > 1 else None
                input_video = input_latent_chunk.tensors[0]
                input_audio = input_latent_chunk.tensors[1] if len(input_latent_chunk.tensors) > 1 else None
                
                combined_video = torch.cat([overlap_video, input_video], dim=2)
                combined_tensors = [combined_video]
                if overlap_audio is not None and input_audio is not None:
                    combined_audio = torch.cat([overlap_audio, input_audio], dim=3)
                    combined_tensors.append(combined_audio)
                
                from comfy.nested_tensor import NestedTensor
                output_latent = {"samples": NestedTensor(combined_tensors)}
                if "type" in overlap_latent:
                    output_latent["type"] = overlap_latent["type"]
            elif overlap_latent is not None:
                # Regular tensor
                output_latent = {"samples": torch.cat([overlap_latent["samples"], input_latent_chunk], dim=2)}
                if "type" in overlap_latent:
                    output_latent["type"] = overlap_latent["type"]
        elif input_latent_chunk is not None:
            # First chunk, no overlap
            latent_dict = latent if isinstance(latent, dict) else {"samples": latent}
            output_latent = {"samples": input_latent_chunk}
            if "type" in latent_dict:
                output_latent["type"] = latent_dict["type"]
        elif overlap_latent is not None:
            # Only overlap (no input latent for this chunk)
            output_latent = overlap_latent

        # finalise out images, resize and concat together
        out_images_torch = None
        if len(out_images) > 0:
            out_images_resized = list(
                map(lambda tensor: resize_image(tensor, w, h, pad=True), out_images))
            out_images_torch = torch.cat(out_images_resized)

        # finalise out masks, resize and concat together
        out_masks_torch = None
        if len(out_masks) > 0:
            out_masks_resized = list(
                map(lambda tensor: resize_mask(tensor, w, h, pad=True), out_masks))
            out_masks_torch = torch.cat(out_masks_resized)

        # finalise out audio, concat together
        out_audio_dict = None
        if len(out_audio) > 0:
            out_audio_dict = concat_audios(out_audio)

        chunker_data = {
            "start_node_id": self.hidden.unique_id,
            "index": s["index"],
            "chunker_config": c,
            "chunk_lengths": chunk_lengths,
            "overlap_latent_count": overlap_latent_count,
            "audio_latent_overlap_count": audio_overlap_count,
            "fps": out_fps,
            "is_i2v": out_images_torch is not None and len(out_images_torch) > 0,
            "ts_chunk_starts": [
                *s["ts_chunk_starts"],
                ts_chunk_start,
            ],
        }

        ui_values = {
            "input_label_values": {
                "video": format_video(video),
                "images": format_images(images),
                "masks": format_masks(masks),
                "audio": format_audio(audio),
                "fps": format_fps(fps),
                "latent": format_latent(latent),
            },
            "output_label_values": {
                "images": format_images(out_images_torch),
                "masks": format_masks(out_masks_torch),
                "audio": format_audio(out_audio_dict),
                "latent": format_latent(output_latent),
            },
        }

        return io.NodeOutput(
            chunker_data,
            out_images_torch,
            out_masks_torch,
            out_audio_dict,
            output_latent,
            ui={"values": [ui_values]},
        )
