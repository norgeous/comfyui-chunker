from ..lib.utils import log
from ..lib.utils_av import save, load, mux
from ..lib.utils_tensor import resize_mask
from ..lib.utils_image_text_overlay import create_preview_video
from ..lib.utils_comfy import comfyui_repeat_nodes, get_node_ids_by_type
from comfy_execution.graph_utils import GraphBuilder
from ..lib.utils_format import format_images, format_masks, format_audio, format_latents, format_fps, format_milliseconds
from ..lib.utils_performance import get_ts, predict

class ChunkerCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunker_data": ("CHUNKER_DATA", {"tooltip": "Connect chunker_data from Chunker node to here"}),
                # "show_debug": ("BOOLEAN", {"default": True, "tooltip": "Show debug overlay in preview"}),
                "select_overlaps_from": (["this_chunk", "previous_chunk"], {"default": "this_chunk", "tooltip": "When combining, select the overlaps from current or previous chunk"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Processed chunk of images"}),
                "masks": ("MASK", {"tooltip": "Processed chunk of masks"}),
                "audio": ("AUDIO", {"tooltip": "Processed chunk of audio"}),
                "latents": ("LATENT", {"tooltip": "Latents"}),
                "store": ("*",), # hidden by js
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "masks", "audio", "fps")
    OUTPUT_TOOLTIPS = (
        "Combined images from all chunks",
        "Combined masks from all chunks",
        "Combined audio from all chunks",
        "FPS",
    )
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    DESCRIPTION = "ChunkerCombine"
    OUTPUT_NODE = True

    def execute(
        self,
        chunker_data,
        # show_debug,
        select_overlaps_from,
        images=None,
        masks=None,
        audio=None,
        latents=None,
        store=None,
        dynprompt=None,
        unique_id=None,
    ):
        if images is None and masks is None and audio is None:
            raise ValueError("At least one of images, masks, or audio must be provided.")

        d = chunker_data
        c = d["chunker_config"]
        s = store if store is not None else {
            "chunks": [],
            "preview_chunks": [],
            "ts_chunk_ends": [],
            "last_latents": None,
        }

        # lanczos resize masks to match images size
        if images is not None and masks is not None:
            masks = resize_mask(masks, images.shape[2], images.shape[1])

        # save input images, masks and / or audio to lossless file
        ts = get_ts()
        log("Save chunk...", end="")
        chunk_path = save(
            images=images,
            masks=masks,
            audio=audio,
            fps=d["fps"],
            profile="webm",
            # profile="mp4",
            alpha_mode="2ndStream",
            filename_prefix="video/chunker/tmp/chunk",
        )[0]
        s["chunks"].append(chunk_path)
        print(f"done ({format_milliseconds(get_ts() - ts)}) {chunk_path}")

        # Make preview from inputs
        ts = get_ts()
        log("Make preview...", end="")
        preview = create_preview_video(images, masks, True, d, c)
        print(f"done ({format_milliseconds(get_ts() - ts)})")

        # Save preview
        ts = get_ts()
        log("Save preview...", end="")
        preview_path = save(
            images=preview,
            masks=None,
            audio=audio,
            fps=d["fps"],
            profile="webm",
            # profile="mp4",
            filename_prefix="video/chunker/tmp/preview",
        )[0]
        s["preview_chunks"].append(preview_path)
        print(f"done ({format_milliseconds(get_ts() - ts)}) {preview_path}")

        # combine all preview chunks to a new file, excluding the overlaps
        ts = get_ts()
        log("Mux all previews...", end="")
        all_preview_path, all_preview_frontend_data = mux(
            paths=s["preview_chunks"],
            filename_prefix="video/chunker/tmp/all-preview",
            overlap=c["chunk_overlap"],
            select_overlaps_from=select_overlaps_from,
        )
        print(f"done ({format_milliseconds(get_ts() - ts)}) {all_preview_path}")

        # figure out if we have completed all chunks
        is_done = d["index"] + 1 >= c["chunk_count"]

        # if no more chunks needed, return early
        if is_done:
            ts = get_ts()
            log("Mux all chunks...", end="")
            out_path = mux(
                paths=s["chunks"],
                filename_prefix="video/chunker/final",
                overlap=c["chunk_overlap"],
                select_overlaps_from=select_overlaps_from,
            )[0]
            print(f"done ({format_milliseconds(get_ts() - ts)})")
 
            ts = get_ts()
            log("Load final tensors...", end="")
            out_images_torch, out_masks_torch, out_audio_dict, fps = load(path=out_path, alpha_mode="2ndStream")
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
                    "latents": format_latents(latents),
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
        graph = GraphBuilder()
        comfyui_repeat_nodes(dynprompt, graph, unique_id, d["start_node_id"])

        # update the store in the new_divide
        new_divide = graph.lookup_node(d["start_node_id"])
        new_divide.set_input("store", {
            "index": d["index"] + 1,
            "last_chunk_path": s["chunks"][-1] if len(s["chunks"]) > 0 else None, # filename of last chunk saved
            "ts_chunk_starts": d["ts_chunk_starts"],
            "last_latents": latents,
        })

        # increment seeds in cloned KSampler nodes, to prevent same motion in each chunk (for Wan21)
        ids = get_node_ids_by_type(graph.finalize(), "KSampler")
        for id in ids:
            real_id = id.replace(f"{unique_id}.0.0.", "")
            node = graph.lookup_node(real_id)
            seed = node.get_input("seed")
            node.set_input("seed", seed + d["index"] + 1)

        # increment seeds in cloned KSamplerAdvanced nodes, to prevent same motion in each chunk (for Wan22)
        ids = get_node_ids_by_type(graph.finalize(), "KSamplerAdvanced")
        for id in ids:
            real_id = id.replace(f"{unique_id}.0.0.", "")
            node = graph.lookup_node(real_id)
            seed = node.get_input("noise_seed")
            node.set_input("noise_seed", seed + d["index"] + 1)

        # increment seeds in cloned RandomNoise nodes, to prevent same motion in each chunk (for Wan22)
        ids = get_node_ids_by_type(graph.finalize(), "RandomNoise")
        for id in ids:
            real_id = id.replace(f"{unique_id}.0.0.", "")
            node = graph.lookup_node(real_id)
            seed = node.get_input("noise_seed")
            node.set_input("noise_seed", seed + d["index"] + 1)

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
                "latents": format_latents(latents),
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

        return {
            "ui": {"values": [ui_values]},
            "result": (
                new_combine.out(0), # images
                new_combine.out(1), # masks
                new_combine.out(2), # audio
                new_combine.out(3), # fps
            ),
            "expand": graph.finalize(),
        }
