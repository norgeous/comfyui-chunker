# import os
from comfy_api.latest import io
from ..lib.utils import log
from ..lib.av_save import av_save, Profile
from ..lib.av_load import av_load
from ..lib.av_combine import av_combine, BlendMode
from ..lib.utils_tensor import resize_mask, mask_to_image, image_to_mask
from ..lib.create_preview_video import create_preview_video
from ..lib.utils_comfy import get_next_save_path
from ..lib.utils_comfy_repeat_nodes import get_clone_ids, comfyui_repeat_nodes, increment_all_seeds
from ..lib.utils_format import format_images, format_masks, format_audio, format_fps, format_milliseconds
from ..lib.utils_performance import get_ts

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
                    options=list(map(lambda member: member.value, BlendMode)),
                    default=BlendMode.NEWER_ONLY.value,
                    tooltip="When chunk_overlap is more than zero this setting determines how images and audio are blended",
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
            "ts_chunk_ends": [],
            "last_all_preview": None,
        }

        # lanczos resize masks to match images size
        if images is not None and masks is not None:
            masks = resize_mask(masks, images.shape[2], images.shape[1])

        # save input images and audio to lossless file
        ts = get_ts()
        log("Save chunk...", end="")
        chunk_path = get_next_save_path("chunker-chunk", "mp4")[0]
        chunk_path = av_save(
            images=images,
            masks=masks,
            audio=audio,
            fps=d["fps"],
            output_path=chunk_path,
        )
        s["chunks"].append(chunk_path)
        print(f"done ({format_milliseconds(get_ts() - ts)})")

        # Make preview from inputs
        ts = get_ts()
        log("Make preview images...", end="")
        preview = create_preview_video(images, masks, audio, d, c, overlap_blend_mode)
        print(f"done ({format_milliseconds(get_ts() - ts)})")

        # Save preview
        ts = get_ts()
        log("Save preview...", end="")
        preview_path = get_next_save_path("chunker-preview", "webm")[0]
        preview_path = av_save(
            images=preview,
            masks=masks,
            audio=audio,
            fps=d["fps"],
            output_path=preview_path,
            profile=Profile.WEB,
        )
        print(f"done ({format_milliseconds(get_ts() - ts)})")

        # combine all preview chunks to a new file, blending the overlaps
        ts = get_ts()
        log("Combine all previews...", end="")
        all_preview_path, all_preview_frontend_data = get_next_save_path("chunker-preview-all", "webm")
        av_combine(
            paths=[s["last_all_preview"], preview_path] if s["last_all_preview"] is not None else [preview_path],
            output_path=all_preview_path,
            overlap_frame_count=c["chunk_overlap"],
            video_blend_mode=BlendMode(overlap_blend_mode),
            audio_blend_mode=BlendMode(overlap_blend_mode),
            profile=Profile.WEB,
        )
        s["last_all_preview"] = all_preview_path
        print(f"done ({format_milliseconds(get_ts() - ts)})")

        # figure out if we have completed all chunks
        is_done = d["index"] + 1 >= c["chunk_count"]

        # if no more chunks needed, return early
        if is_done:
            ts = get_ts()
            log("Combine all chunks...", end="")
            final_path = get_next_save_path("chunker-chunk-all", "mp4")[0]
            av_combine(
                paths=s["chunks"],
                output_path=final_path,
                overlap_frame_count=c["chunk_overlap"],
                video_blend_mode=BlendMode(overlap_blend_mode),
                audio_blend_mode=BlendMode(overlap_blend_mode),
            )
            print(f"done ({format_milliseconds(get_ts() - ts)})")

            out_images_torch, out_masks_torch, out_audio_dict = None, None, None
            ts = get_ts()
            log("Load final tensors...", end="")
            out_images_torch, out_masks_torch, out_audio_dict, _ = av_load(path=final_path)
            print(f"done ({format_milliseconds(get_ts() - ts)})")

            ts_chunk_end = get_ts()
            s["ts_chunk_ends"] = [
                *s["ts_chunk_ends"],
                ts_chunk_end,
            ]

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
                "ts_chunk_starts": d["ts_chunk_starts"],
                "ts_chunk_ends": s["ts_chunk_ends"],
            }

            log(f"Finished all chunks {d['index'] + 1} of {c['chunk_count']}!")

            return {
                "ui": {"values": [ui_values]},
                "result":(
                    out_images_torch,
                    out_masks_torch,
                    out_audio_dict,
                    float(d["fps"]), # it might be a Fraction or float, so cast to float
                )
            }

        # clone all the nodes between ChunkerDivide and ChunkerCombine
        ts = get_ts()
        clone_ids = get_clone_ids(self.hidden.dynprompt, d["start_node_id"], self.hidden.unique_id, ["Noise"] if increment_seeds else [])
        log(f"Cloning {len(clone_ids)} nodes for next chunk; ", end="")
        id_labels = list(map(lambda id: int(self.hidden.dynprompt.get_display_node_id(id)), clone_ids))
        id_labels.sort()
        print(f"{', '.join(list(map(lambda id: f'#{id}', id_labels)))}...", end="")
        graph = comfyui_repeat_nodes(self.hidden.dynprompt, clone_ids, self.hidden.unique_id)

        # update the store in the new_divide
        new_divide = graph.lookup_node(d["start_node_id"])
        new_divide.set_input("store", {
            "index": d["index"] + 1,
            "last_chunk_path": s["chunks"][-1] if len(s["chunks"]) > 0 else None, # filename of last chunk saved
            "ts_chunk_starts": d["ts_chunk_starts"],
        })

        # update the store in the new_combine (this node)
        new_combine = graph.lookup_node("Recurse")
        ts_chunk_end = get_ts()
        s["ts_chunk_ends"] = [
            *s["ts_chunk_ends"],
            ts_chunk_end,
        ]
        new_combine.set_input("store", s)

        print(f"done ({format_milliseconds(get_ts() - ts)})")

        # Increment seed in cloned nodes with "Sampler" or "Noise" in the node type name, such as;
        # - KSampler
        # - KSamplerAdvanced
        # - RandomNoise (used by SamplerCustomAdvanced)
        # - ClownsharKSampler
        # this is to prevent same motion in each chunk (when using Wan or LTX)
        if increment_seeds: increment_all_seeds(graph, self.hidden.unique_id)

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
            "ts_chunk_starts": d["ts_chunk_starts"],
            "ts_chunk_ends": s["ts_chunk_ends"],
        }

        log(f"Finished chunk {d['index'] + 1} of {c['chunk_count']}")

        return io.NodeOutput(
            new_combine.out(0), # images
            new_combine.out(1), # masks
            new_combine.out(2), # audio
            new_combine.out(3), # fps
            ui={"values": [ui_values]},
            expand=graph.finalize(),
        )
