import torch
import math
import safetensors.torch
from comfy_api.latest import io
from ..lib.utils import count, log
from ..lib.plan_chunks import plan_chunks
from ..lib.av_load import av_load
from ..lib.utils_comfy import concat_audios, stretch_audio_to_fps
from ..lib.utils_tensor import resize_image, resize_mask
from ..lib.utils_format import (format_images, format_masks, format_audio, format_fps, format_video, format_latent)
from ..lib.utils_latent_combine import build_overlap_noise_mask
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
        "chunk_length_settings": {"default": 100, "min": 1, "step": 1},   # n
        "length_to_video_latent_length": lambda length: (length + 3) // 4,
        "length_to_audio_latent_length": lambda length: 0,
    },
    Mode.WAN2: {
        "dimension_adjuster": lambda length: (length // 16) * 16, # 16n
        "length_adjuster": lambda length: (math.ceil((length - 1) / 4) * 4) + 1, # 4n+1. example: 1, 5, 9, 13, 17
        "fps": 16.0,
        "chunk_length_settings": {"default": 81, "min": 1, "step": 4},   # 4n+1
        "length_to_video_latent_length": lambda length: (length + 3) // 4,
        "length_to_audio_latent_length": lambda length: 0,
    },
    Mode.LTX2: {
        "dimension_adjuster": lambda length: (length // 32) * 32, # 32n
        "length_adjuster": lambda length: (math.ceil((length - 1) / 8) * 8) + 1, # 8n+1. example: 1, 9, 17, 25, 33
        "fps": 25.0,
        "chunk_length_settings": {"default": 81, "min": 1, "step": 8},   # 8n+1
        "length_to_video_latent_length": lambda length: (length + 7) // 8,
        "length_to_audio_latent_length": lambda length: length,
    },
    Mode.MINIMAX_H3: {
        "dimension_adjuster": lambda length: (length // 32) * 32, # 32n
        "length_adjuster": lambda length: (math.ceil((length - 5) / 17) * 17) + 5, # 17n+5. example: 5, 22, 39, 56, 73
        "fps": 24.0,
        "chunk_length_settings": {"default": 107, "min": 5, "step": 17},  # 17n+5
        "length_to_video_latent_length": lambda length: (length // 17) * 5 + ((length % 17) + 3) // 4, # 5n+2
        "length_to_audio_latent_length": lambda length: 28 * (length // 17) + round((length % 17) * 28 / 17) + math.ceil((length // 17) / 3), # 28n+ceil(n/3)+8
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
                io.Latent.Input(
                    "latent",
                    optional=True,
                    tooltip="latent (optional)",
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
                    tooltip="audio (optional). Time-stretched to the mode's fps before chunking when a source fps is known",
                ),
                io.Float.Input(
                    "original_fps",
                    optional=True,
                    force_input=True,
                    tooltip="Passthrough: the FPS of the input video, or a manual override. Not used for processing; the mode's FPS is used.",
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
                                    **mode_settings[member]["chunk_length_settings"],
                                ),
                                *(
                                    [
                                        io.Vae.Input(
                                            "video_vae",
                                            optional=True,
                                            tooltip="MiniMax H3 video VAE, used to encode reference frames into the packed latent",
                                        ),
                                        io.Vae.Input(
                                            "audio_vae",
                                            optional=True,
                                            tooltip="MiniMax H3 audio VAE, used to encode reference audio into the packed latent",
                                        ),
                                    ]
                                    if member == Mode.MINIMAX_H3
                                    else []
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
                io.Int.Input(
                    "width",
                    optional=True,
                    force_input=True,
                    default=0,
                    min=0,
                    max=8192,
                    step=1,
                    tooltip="Target chunk image width. 0 = derive from input. Rounded by the mode's dimension_adjuster.",
                ),
                io.Int.Input(
                    "height",
                    optional=True,
                    force_input=True,
                    default=0,
                    min=0,
                    max=8192,
                    step=1,
                    tooltip="Target chunk image height. 0 = derive from input. Rounded by the mode's dimension_adjuster.",
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
                io.Latent.Output(
                    "latent",
                    tooltip="Chunk of latent. First chunk is from input, subsequent chunks contain overlap + input",
                ),
                io.Image.Output(
                    "images",
                    tooltip="Chunk of images. First chunk is from input, subsequent chunks contain overlap + input",
                ),
                io.Mask.Output(
                    "masks",
                    tooltip="Chunk of masks. First chunk is from input, subsequent chunks contain overlap + input",
                ),
                io.Audio.Output(
                    "audio",
                    tooltip="Chunk of audio. First chunk is from input, subsequent chunks contain overlap + input",
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
        original_fps=None,
        width=None,
        height=None,
        store=None,
    ) -> io.NodeOutput:
        ts_chunk_start = get_ts()

        selected_mode = Mode(mode["mode"])
        settings = mode_settings[selected_mode]
        chunk_length = mode["chunk_length"]
        video_vae = mode.get("video_vae")
        audio_vae = mode.get("audio_vae")

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

        source_fps = original_fps if original_fps is not None else video_fps

        # Fit input audio to the mode's fps so chunk slicing stays aligned
        if audio is not None and source_fps:
            audio = stretch_audio_to_fps(audio, source_fps, settings["fps"])

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
                audio["waveform"].shape[-1] // math.floor(audio["sample_rate"] / settings["fps"]) if audio is not None else 0,
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

        w = settings["dimension_adjuster"](width) if width else None
        h = settings["dimension_adjuster"](height) if height else None

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

        # Compute the latent token counts in this chunk's overlap region from the
        # pixel layout. Needed to trim the head of a sliced or VAE-encoded chunk and
        # for Combine to dedup overlap tokens on recombination.
        video_overlap_latent_count = 0
        video_overlap_count = 0
        audio_overlap_count = 0
        audio_overlap_start = 0
        audio_overlap_end = 0
        if s["index"] > 0 and overlap_length > 0:
            length_to_video = settings["length_to_video_latent_length"]
            length_to_audio = settings["length_to_audio_latent_length"]
            overlap_pixel_start = max(0, start - overlap_length)
            overlap_pixel_end = start
            video_overlap_start, video_overlap_end = length_to_video(overlap_pixel_start), length_to_video(overlap_pixel_end)
            audio_overlap_start, audio_overlap_end = length_to_audio(overlap_pixel_start), length_to_audio(overlap_pixel_end)

            # H3 VAE uses token_overlap=2 for 5-frame overlap
            token_overlap = settings.get("token_overlap", 2)
            if video_overlap_end - video_overlap_start > token_overlap:
                video_overlap_start = video_overlap_end - token_overlap

            video_overlap_count = video_overlap_end - video_overlap_start
            audio_overlap_count = audio_overlap_end - audio_overlap_start

        # load latent overlap from previous chunk's safetensors
        overlap_latent = None
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
                    length_to_audio = settings["length_to_audio_latent_length"]
                    has_audio = length_to_audio(c["total_length"]) > 0

                    # Slice overlap from the end of previous chunk's latent
                    if hasattr(full_latent_tensor, "tensors"):  # NestedTensor
                        video_t = full_latent_tensor.tensors[0]  # [B, 24, T, H, W]
                        audio_t = full_latent_tensor.tensors[1]  # [B, 32, 2, T]

                        file_t = video_t.shape[2]
                        video_overlap_start = file_t - video_overlap_count
                        video_overlap_end = file_t
                        video_overlap = video_t[:, :, video_overlap_start:video_overlap_end, :, :]
                        if has_audio:
                            audio_file_t = audio_t.shape[3]
                            audio_overlap_start = audio_file_t - audio_overlap_count
                            audio_overlap_end = audio_file_t
                        audio_overlap = audio_t[:, :, :, audio_overlap_start:audio_overlap_end] if has_audio else None

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
                        if hasattr(overlap_latent_tensor, "tensors"):
                            video_overlap_latent_count = overlap_latent_tensor.tensors[0].shape[2]
                        else:
                            video_overlap_latent_count = overlap_latent_tensor.shape[2]

        out_images = []
        out_masks = []
        out_audio = []

        # get the overlap from the last chunk (video file) that Combine saved
        if s["last_chunk_path"] is not None and overlap_length > 0:
            overlap_images, overlap_masks, overlap_audio_dict, _ = av_load(
                path=s["last_chunk_path"],
                start=-overlap_length,
            )
            if w is None:
                w = overlap_images.shape[2]
            if h is None:
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
                    if source_fps:
                        video_audio_dict = stretch_audio_to_fps(video_audio_dict, source_fps, settings["fps"])
                    out_audio.append(video_audio_dict)
                if original_fps is None and source_fps is None and loaded_fps:
                    source_fps = loaded_fps

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
            samples_per_frame = math.floor(audio["sample_rate"] / settings["fps"])
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

        # prepare chunk of latent from input
        input_latent_chunk = None
        if latent is not None:
            latent_dict = latent if isinstance(latent, dict) else {"samples": latent}
            full_input_latent = latent_dict["samples"]

            # Compute video and audio latent ranges from mode settings
            length_to_video = settings["length_to_video_latent_length"]
            length_to_audio = settings["length_to_audio_latent_length"]
            has_audio = length_to_audio(c["total_length"]) > 0
            video_latent_start, video_latent_end = length_to_video(start), length_to_video(end)
            audio_latent_start, audio_latent_end = length_to_audio(start), length_to_audio(end)

            if hasattr(full_input_latent, "tensors"):  # NestedTensor
                video_t = full_input_latent.tensors[0]  # [B, 24, T, H, W]
                audio_t = full_input_latent.tensors[1]  # [B, 32, 2, T]

                video_chunk = video_t[:, :, video_latent_start + video_overlap_latent_count:video_latent_end, :, :]
                audio_chunk = audio_t[:, :, :, audio_latent_start + audio_overlap_count:audio_latent_end] if has_audio else None

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
                input_latent_chunk = full_input_latent[:, :, video_latent_start + video_overlap_latent_count:video_latent_end, :, :]
        elif (
            video_vae is not None
            and out_images_torch is not None
            and (out_audio_dict is None or audio_vae is not None)
        ):
            # No pre-encoded latent: VAE-encode this chunk's frames (+ audio) into a latent.
            from ..lib.av_encode import encode_video, encode_audio, pack_av_latent

            log(f"ChunkerRepeat#{self.hidden.dynprompt.get_display_node_id(self.hidden.unique_id)}: VAE-encoding chunk {s['index'] + 1} of {c['chunk_count']}...")

            # out_images_torch holds overlap + new frames; trim the encoded head by the
            # overlap token count so the (denoised) overlap from the previous chunk can
            # be prepended below instead.
            encoded_video = encode_video(video_vae, out_images_torch)
            if video_overlap_count > 0 and encoded_video.shape[2] > video_overlap_count:
                encoded_video = encoded_video[:, :, video_overlap_count:, :, :]

            if audio_vae is not None and out_audio_dict is not None:
                encoded_audio = encode_audio(audio_vae, out_audio_dict)
                if audio_overlap_count > 0 and encoded_audio.shape[3] > audio_overlap_count:
                    encoded_audio = encoded_audio[:, :, :, audio_overlap_count:]
                input_latent_chunk = pack_av_latent(encoded_video, encoded_audio)
            else:
                input_latent_chunk = encoded_video
        elif video_vae is not None and out_images_torch is not None and out_audio_dict is not None and audio_vae is None:
            log(f"ChunkerRepeat#{self.hidden.dynprompt.get_display_node_id(self.hidden.unique_id)}: Skipping VAE encode for chunk {s['index'] + 1}; audio present without an audio_vae")

        # Combine overlap_latent + input_latent_chunk
        output_latent = None
        if overlap_latent is not None and input_latent_chunk is not None:
            # Prepend overlap to input chunk
            if hasattr(overlap_latent["samples"], "tensors") and hasattr(input_latent_chunk, "tensors"):
                # Both NestedTensor — find video (5D) and audio (4D) by dimension
                overlap_video = overlap_audio = None
                for t in overlap_latent["samples"].tensors:
                    if t.dim() == 5:
                        overlap_video = t
                    elif t.dim() == 4:
                        overlap_audio = t
                input_video = input_audio = None
                for t in input_latent_chunk.tensors:
                    if t.dim() == 5:
                        input_video = t
                    elif t.dim() == 4:
                        input_audio = t

                combined_video = None
                if overlap_video is not None and input_video is not None:
                    combined_video = torch.cat([overlap_video, input_video], dim=2)
                elif overlap_video is not None:
                    combined_video = overlap_video
                elif input_video is not None:
                    combined_video = input_video

                combined_tensors = []
                if combined_video is not None:
                    combined_tensors.append(combined_video)
                if overlap_audio is not None and input_audio is not None:
                    combined_audio = torch.cat([overlap_audio, input_audio], dim=3)
                    combined_tensors.append(combined_audio)
                elif overlap_audio is not None:
                    combined_tensors.append(overlap_audio)
                elif input_audio is not None:
                    combined_tensors.append(input_audio)

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

        # Pin the preserved overlap region so a partial-denoise pass does not rewrite it.
        # The mask ramps linearly from 0 at the overlap start to 1 by the overlap end, then
        # stays 1 over the new frames. Only meaningful for H3's packed NestedTensor latent.
        if (output_latent is not None and c["mode"] == "minimax-h3"
                and video_overlap_latent_count > 0
                and hasattr(output_latent["samples"], "tensors")):
            overlap_out_video = output_latent["samples"].tensors[0]
            overlap_out_audio = output_latent["samples"].tensors[1] if len(output_latent["samples"].tensors) > 1 else None
            if overlap_out_audio is not None:
                output_latent["noise_mask"] = build_overlap_noise_mask(
                    overlap_out_video, overlap_out_audio, video_overlap_latent_count, audio_overlap_count)

        chunker_data = {
            "start_node_id": self.hidden.unique_id,
            "index": s["index"],
            "chunker_config": c,
            "chunk_lengths": chunk_lengths,
            "video_vae": video_vae,
            "audio_vae": audio_vae,
            "video_overlap_latent_count": video_overlap_latent_count,
            "audio_overlap_latent_count": audio_overlap_count,
            "original_fps": source_fps,
            "fps": settings["fps"],
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
                "original_fps": format_fps(original_fps),
                "latent": format_latent(latent),
                "width": str(width) if width else "\u2205",
                "height": str(height) if height else "\u2205",
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
            output_latent,
            out_images_torch,
            out_masks_torch,
            out_audio_dict,
            ui={"values": [ui_values]},
        )
