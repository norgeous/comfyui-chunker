import os
from comfy_api.latest import io
from ..lib.utils import log
# from ..lib.utils_av import save, load #, mux, mux2
# from ..lib.mux import mux
# from ..lib.utils_av_combine import combine, BlendMode
from ..lib.utils_av import av_save, av_load, av_combine, BlendMode
from ..lib.utils_tensor import resize_mask
from ..lib.utils_image_text_overlay import create_preview_video
from ..lib.utils_comfy import comfyui_repeat_nodes, increment_all_seeds, get_next_save_path
from ..lib.utils_format import format_images, format_masks, format_audio, format_fps, format_milliseconds
from ..lib.utils_performance import get_ts, predict
# from ..enum.options import OverlapBlendModes

class ChunkerCombine(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ChunkerCombine",
            display_name="\U0001F36B Combine",
            category="chunker",
            inputs=[
                io.Custom("CHUNKER_DATA").Input("chunker_data",
                    tooltip="Connect chunker_data from ChunkerDivide node to here"             
                ),
                io.Image.Input("images",
                    optional=True,
                    tooltip="Processed chunk of images",
                ),
                io.Mask.Input("masks",
                    optional=True,
                    tooltip="Processed chunk of masks",
                ),
                io.Audio.Input("audio",
                    optional=True,
                    tooltip="Processed chunk of audio",
                ),
                io.Combo.Input("overlap_blend_mode",
                    options=list(map(lambda member: member.name, BlendMode)),
                    default=BlendMode.NEWER_ONLY.name,
                    tooltip="When combining and chunk_overlap is more than zero, select the overlaps from current or previous chunk",
                ),
                io.Boolean.Input("increment_seeds",
                    tooltip="Increment \"seed\" or \"noise_seed\" inputs in cloned nodes that have \"Sampler\" or \"Noise\" within the node type name",
                    default=True,
                ),
                io.Custom("*").Input("store",
                    optional=True,
                ),
            ],
            outputs=[
                io.Image.Output("images",
                    tooltip="Combined images from all chunks",
                ),
                io.Mask.Output("masks",
                    tooltip="Combined masks from all chunks",
                ),
                io.Audio.Output("audio",
                    tooltip="Combined audio from all chunks",
                ),
                io.Float.Output("fps",
                    tooltip="FPS",
                ),
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.dynprompt],
            is_output_node=True,
            enable_expand=True,
        )

    @classmethod
    def execute(
        self,
        chunker_data,
        overlap_blend_mode,
        increment_seeds,
        images=None,
        masks=None,
        audio=None,
        store=None,
    ):
        if images is None and masks is None and audio is None:
            raise ValueError("At least one of images, masks, or audio must be provided.")

        d = chunker_data
        c = d["chunker_config"]
        s = store if store is not None else {
            "chunks": [],
            "preview_chunks": [],
            "ts_chunk_ends": [],
            "last_all_preview": None,
        }

        # lanczos resize masks to match images size
        if images is not None and masks is not None:
            masks = resize_mask(masks, images.shape[2], images.shape[1])

        # save input images, masks and / or audio to lossless file
        ts = get_ts()
        log("Save chunk...", end="")
        chunk_path = get_next_save_path("video/chunker/tmp/chunk", "mp4")[0]
        chunk_path = av_save(
            images=images,
            # masks=masks,
            audio=audio,
            fps=d["fps"],
            # profile="mp4",
            # alpha_mode="2ndStream",
            output_path=chunk_path,
        )
        s["chunks"].append(chunk_path)
        print(f"done ({format_milliseconds(get_ts() - ts)})")

        # Make preview from inputs
        ts = get_ts()
        log("Make preview...", end="")
        preview = create_preview_video(images, masks, True, d, c)
        print(f"done ({format_milliseconds(get_ts() - ts)})")

        # Save preview
        ts = get_ts()
        log("Save preview...", end="")
        preview_path = get_next_save_path("video/chunker/tmp/preview", "mp4")[0]
        preview_path = av_save(
            images=preview,
            # masks=None,
            audio=audio,
            fps=d["fps"],
            # profile="mp4",
            output_path=preview_path,
        )
        s["preview_chunks"].append(preview_path)
        print(f"done ({format_milliseconds(get_ts() - ts)})")

        # combine all preview chunks to a new file, excluding the overlaps
        ts = get_ts()
        log("Combine all previews...", end="")
        if s["last_all_preview"] is not None and os.path.exists(s["last_all_preview"]):
            os.remove(s["last_all_preview"]) 
        all_preview_path, all_preview_frontend_data = get_next_save_path("video/chunker/tmp/all-preview", "mp4")
        av_combine(
            paths=s["preview_chunks"],
            output_path=all_preview_path,
            overlap_frame_count=c["chunk_overlap"],
            video_blend_mode=BlendMode[overlap_blend_mode],
            audio_blend_mode=BlendMode[overlap_blend_mode],
        )
        s["last_all_preview"] = all_preview_path
        print(f"done ({format_milliseconds(get_ts() - ts)})")

        # figure out if we have completed all chunks
        is_done = d["index"] + 1 >= c["chunk_count"]

        # if no more chunks needed, return early
        if is_done:
            ts = get_ts()
            log("Mux all chunks...", end="")
            final_path = get_next_save_path("video/chunker/final", "mp4")[0]
            av_combine(
                paths=s["chunks"],
                output_path=final_path,
                overlap_frame_count=c["chunk_overlap"],
                video_blend_mode=BlendMode[overlap_blend_mode],
                audio_blend_mode=BlendMode[overlap_blend_mode],
            )
            print(f"done ({format_milliseconds(get_ts() - ts)})")

            ts = get_ts()
            log("Delete all temp chunks and preview chunks...", end="")
            for path in [*s["chunks"], *s["preview_chunks"]]:
                if os.path.exists(path):
                    os.remove(path)
            print(f"done ({format_milliseconds(get_ts() - ts)})")

            ts = get_ts()
            log("Load final tensors...", end="")
            out_images_torch, out_audio_dict = av_load(path=final_path)
            out_masks_torch = None
            print(f"done ({format_milliseconds(get_ts() - ts)})")

            ts_chunk_end = get_ts()
            s["ts_chunk_ends"] = [
                *s["ts_chunk_ends"],
                ts_chunk_end,
            ]
            historical_deltas = [e - s for s, e in zip(d["ts_chunk_starts"], s["ts_chunk_ends"])]

            ui_values = {
                "input_label_values": {
                    "images": format_images(images),
                    "masks": format_masks(masks),
                    "audio": format_audio(audio),
                },
                "output_label_values": {
                    "images": format_images(out_images_torch),
                    "masks": format_masks(out_masks_torch),
                    "audio": format_audio(out_audio_dict),
                    "fps": format_fps(d["fps"]),
                },
                "index": d["index"],
                "chunk_count": c["chunk_count"],
                "video_path": all_preview_frontend_data,
                "historical_deltas": historical_deltas,
                "predicted_deltas": [],
            }

            log(f"Finished all chunks {d["index"] + 1} of {c["chunk_count"]}!")

            return {
                "ui": {"values": [ui_values]},
                "result":(
                    out_images_torch,
                    out_masks_torch,
                    out_audio_dict,
                    float(d["fps"]), # it might be a Fraction or float, so cast to float
                )
            }

        ts = get_ts()
        log("Cloning nodes for next chunk...", end="")

        # clone all the nodes between Chunker and ChunkerCombine
        graph = comfyui_repeat_nodes(self.hidden.dynprompt, self.hidden.unique_id, d["start_node_id"])

        # update the store in the new_divide
        new_divide = graph.lookup_node(d["start_node_id"])
        new_divide.set_input("store", {
            "index": d["index"] + 1,
            "last_chunk_path": s["chunks"][-1] if len(s["chunks"]) > 0 else None, # filename of last chunk saved
            "ts_chunk_starts": d["ts_chunk_starts"],
        })

        # Increment seed in cloned nodes with "Sampler" or "Noise" in the node type name, such as;
        # - KSampler
        # - KSamplerAdvanced
        # - RandomNoise (used by SamplerCustomAdvanced)
        # - ClownsharKSampler
        # this is to prevent same motion in each chunk (for Wan22 or LTX2)
        if increment_seeds:
            increment_all_seeds(graph, self.hidden.unique_id, d["index"] + 1)

        # update the store in the new_combine (this node)
        new_combine = graph.lookup_node("Recurse")
        ts_chunk_end = get_ts()
        s["ts_chunk_ends"] = [
            *s["ts_chunk_ends"],
            ts_chunk_end,
        ]
        new_combine.set_input("store", s)

        print(f"done ({format_milliseconds(get_ts() - ts)})")

        # predict all chunk deltas in milliseconds
        historical_deltas = [e - s for s, e in zip(d["ts_chunk_starts"], s["ts_chunk_ends"])]
        predicted_deltas = predict(historical_deltas, c["chunk_count"] - d["index"] - 1)

        ui_values = {
            "input_label_values": {
                "images": format_images(images),
                "masks": format_masks(masks),
                "audio": format_audio(audio),
            },
            "output_label_values": {
                "images": None,
                "masks": None,
                "audio": None,
                "fps": None,
            },
            "index": d["index"],
            "chunk_count": c["chunk_count"],
            "video_path": all_preview_frontend_data,
            "historical_deltas": historical_deltas,
            "predicted_deltas": predicted_deltas,
        }

        log(f"Finished chunk {d["index"] + 1} of {c["chunk_count"]} ({format_milliseconds(historical_deltas[-1])})")
        
        return io.NodeOutput(
            new_combine.out(0), # images
            new_combine.out(1), # masks
            new_combine.out(2), # audio
            new_combine.out(3), # fps
            ui={"values": [ui_values]},
            expand=graph.finalize(),
        )
