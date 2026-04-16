import torch
import math
from comfy_api.latest import io
from ..lib.utils import count, log, force_wan_length, fix_total_length, get_this_chunk_length
from ..lib.utils_av import av_load
from ..lib.utils_comfy import concat_audios
from ..lib.utils_tensor import monochrome_image, monochrome_mask, resize_image, resize_mask
from ..lib.utils_format import format_images, format_masks, format_audio, format_fps#, format_latents
from ..lib.utils_performance import get_ts
from ..enum.options import DivideMode

class ChunkerDivide(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ChunkerDivide",
            display_name="\U0001F36B Divide",
            category="chunker",
            inputs=[
                io.Image.Input("images",
                    optional=True,
                    tooltip="images",
                ),
                io.Mask.Input("masks",
                    optional=True,
                    tooltip="masks",
                ),
                io.Audio.Input("audio",
                    optional=True,
                    tooltip="audio",
                ),
                io.Float.Input("fps",
                    optional=True,
                    force_input=True,
                    tooltip="FPS",
                ),
                io.Combo.Input("mode",
                    options=list(map(lambda member: member.name, DivideMode)),
                    tooltip="Adjust chunk_length and total_length to match Wan's format (4n+1) or LTX's format.",
                ),
                io.Int.Input("chunk_length",
                    tooltip="Count of images in each chunk",
                    default=81,
                    min=1,
                    max=4096,
                    step=1,
                ),
                io.Int.Input("chunk_overlap",
                    tooltip="Count of images to overlap between chunks",
                    default=4,
                    min=0,
                    max=4096,
                    step=1,
                ),
                io.Int.Input("total_length",
                    tooltip="Minimum count of images in the final output. 0 to use the images length",
                    default=0,
                    min=0,
                    max=10000,
                    step=1,
                ),
                io.Custom("*").Input("store",
                    optional=True,
                ),
            ],
            outputs=[
                io.Custom("CHUNKER_DATA").Output("chunker_data",
                    tooltip="Connect \"chunker_data\" to the \"ChunkerCombine\" node"           
                ),
                io.Image.Output("images",
                    tooltip="Chunk of images",
                ),
                io.Mask.Output("masks",
                    tooltip="Chunk of masks",
                ),
                io.Audio.Output("audio",
                    tooltip="Chunk of audio",
                ),
                io.Float.Output("fps",
                    tooltip="FPS",
                ),
                io.Int.Output("chunk_length",
                    tooltip="Count of images in this chunk",
                ),
                io.Int.Output("chunk_overlap",
                    tooltip="Count of images to overlap between each chunk",
                ),
                io.Int.Output("total_length",
                    tooltip="Total length of output images",
                ),
                io.Int.Output("chunk_count",
                    tooltip="Count of chunks",
                ),
                io.Int.Output("index",
                    tooltip="The current itteration index, ie; 0, 1, 2, ...",
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return str(get_ts()) # force run if cached, so that start timestamp always updates

    @classmethod
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
    ) -> io.NodeOutput:
        ts_chunk_start = get_ts()

        s = store if store is not None else {
            "index": 0,
            "last_chunk_path": None,
            "ts_chunk_starts": [],
        }

        out_fps = fps
        if out_fps is None and mode.startswith("wan"): out_fps = 16.0
        if out_fps is None and mode.startswith("ltx"): out_fps = 25.0
        if out_fps is None: out_fps = 30.0

        if total_length == 0:
            total_length = max(
                len(images) if images is not None else 0,
                len(masks) if masks is not None else 0,
            )

        if mode.startswith("wan"):
            chunk_length = force_wan_length(chunk_length)
            total_length = fix_total_length(total_length, chunk_length, chunk_overlap)

        # todo
        if mode.startswith("ltx"):
            # force 8n+1 chunk_length. example: 1, 9, 17, 25, 33
            # fix total_length
            pass

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
            overlap_images, overlap_audio_dict, _ = av_load(
                path=s["last_chunk_path"],
                overlap_frame_count=-chunk_overlap,
            )
            overlap_masks = None
            w = overlap_images.shape[2]
            h = overlap_images.shape[1]
            if overlap_images is not None:
                out_images.append(overlap_images)
                if mode == "wan_vace":
                    # preserve overlap images with black masks
                    black_mask = monochrome_mask(w, h, 0)
                    out_masks.append(torch.cat([black_mask] * len(overlap_images)))
            if overlap_masks is not None:
                if mode != "wan_vace":
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
            samples_per_frame = math.floor(audio["sample_rate"] / out_fps)
            samples_already_collected = out_audio[0]["waveform"].shape[-1] if len(out_audio) > 0 else 0
            astart = (start * samples_per_frame) + samples_already_collected
            aend = end * samples_per_frame
            out_audio.append({
                "waveform": audio["waveform"][:,:,astart:aend],
                "sample_rate": audio["sample_rate"],
            })

        if w is None: w = 512
        if h is None: h = 512

        # for wan vace
        if mode == "wan_vace":
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
                "fps": format_fps(out_fps),
                "chunk_length": this_chunk_length,
                "chunk_overlap": chunk_overlap,
                "total_length": total_length,
                "chunk_count": chunk_count,
                "index": s["index"],
            },
        }

        return io.NodeOutput(
            chunker_data,
            out_images_torch,
            out_masks_torch,
            out_audio_dict,
            out_fps,
            this_chunk_length,
            chunk_overlap,
            total_length,
            chunk_count,
            s["index"],
            ui={"values": [ui_values]},
        )