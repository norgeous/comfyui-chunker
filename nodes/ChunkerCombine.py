from comfy_execution.graph_utils import GraphBuilder
from ..lib.utils import log, resize_mask
from ..lib.utils_av import save, mux, load
from ..lib.debug_overlay import create_preview_video
from ..lib.repeat_nodes import comfyui_repeat_nodes, get_node_ids_by_type
from ..lib.utils_format import format_audio, format_fps

class ChunkerCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunker_data": ("CHUNKER_DATA", {"tooltip": "Connect chunker_data from Chunker node to here"}),
                "show_debug": ("BOOLEAN", {"default": True, "tooltip": "Show debug overlay in preview"}),
                "select_overlaps_from": (["this_chunk", "previous_chunk"], {"default": "this_chunk", "tooltip": "When combining, select the overlaps from current or previous chunk"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Processed chunk of images"}),
                "masks": ("MASK", {"tooltip": "Processed chunk of masks"}),
                "audio": ("AUDIO", {"tooltip": "Processed chunk of audio"}),
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
        "Audio from images input of Chunker",
        "FPS",
    )
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    DESCRIPTION = "ChunkerCombine"
    OUTPUT_NODE = True

    def execute(
        self,
        chunker_data,
        show_debug,
        select_overlaps_from,
        images=None,
        masks=None,
        audio=None,
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
        }

        # resize masks to match images size
        if images is not None and masks is not None:
            masks = resize_mask(masks, images.shape[2], images.shape[1])

        # save input images, masks and / or audio to lossless file
        log("Save chunk...", end="")
        chunk_path = save(
            images=images,
            masks=masks,
            audio=audio,
            fps=d["fps"],
            profile="lossless",
            filename_prefix="video/chunker/tmp/chunk",
        )[0]
        s["chunks"].append(chunk_path)
        print("done")

        # create and save preview from inputs
        log("Create preview...", end="")
        preview = create_preview_video(images, masks, show_debug, d, c)
        preview_path = save(
            images=preview,
            masks=None,
            audio=audio,
            fps=d["fps"],
            profile="lossless",
            filename_prefix="video/chunker/tmp/preview",
        )[0]
        s["preview_chunks"].append(preview_path)
        print("done")

        # figure out if we have completed all chunks
        is_done = d["index"] + 1 >= c["chunk_count"]

        # combine all preview chunks to a new file, excluding the overlaps
        log("Mux preview...", end="")
        all_preview_frontend_data = mux(
            paths=s["preview_chunks"],
            profile="web-rgba",
            filename_prefix="video/chunker/tmp/preview-web",
            overlap=c["chunk_overlap"],
            select_overlaps_from=select_overlaps_from,
        )[1]
        print("done")

        # if no more chunks needed, return early
        if is_done:
            log("Mux chunks...", end="")
            out_path = mux(
                paths=s["chunks"],
                profile="lossless",
                filename_prefix="video/chunker/final",
                overlap=c["chunk_overlap"],
                select_overlaps_from=select_overlaps_from,
            )[0]
            print("done")
 
            log("Load final tensors...", end="")
            out_images_torch, out_masks_torch, out_audio_dict, fps = load(path=out_path)
            print("done")

            ui_values = {
                "input_label_values": {
                    "images": len(images) if images is not None else 0,
                    "masks": len(masks) if masks is not None else 0,
                    "audio": format_audio(audio),
                },
                "output_label_values": {
                    "images": len(out_images_torch) if out_images_torch is not None else 0,
                    "masks": len(out_masks_torch) if out_masks_torch is not None else 0,
                    "audio": format_audio(out_audio_dict),
                    "fps": format_fps(d["fps"]),
                },
                "index": d["index"],
                "chunk_count": c["chunk_count"],
                "video_path": all_preview_frontend_data,
            }

            log(f"Finished all chunks {d["index"] + 1} of {c["chunk_count"]}!")

            return {
                "ui": {"values": [ui_values]},
                "result":(
                    out_images_torch,
                    out_masks_torch,
                    out_audio_dict,
                    d["fps"],
                )
            }

        # clone all the nodes between Chunker and ChunkerCombine
        graph = GraphBuilder()
        comfyui_repeat_nodes(dynprompt, graph, unique_id, d["start_node_id"])

        # update the store in the new_chunker
        new_chunker = graph.lookup_node(d["start_node_id"])
        new_chunker.set_input("store", {
            "index": d["index"] + 1,
            "last_chunk_path": s["chunks"][-1] if len(s["chunks"]) > 0 else None, # filename of last chunk saved
        })

        # increment seeds in cloned KSamplers, to prevent same motion in each chunk (for Wan)
        ids = get_node_ids_by_type(graph.finalize(), "KSampler")
        for id in ids:
            real_id = id.replace(f"{unique_id}.0.0.", "")
            node = graph.lookup_node(real_id)
            seed = node.get_input("seed")
            node.set_input("seed", seed + d["index"] + 1)

        # increment seeds in cloned KSamplersAdvanced, to prevent same motion in each chunk (for Wan)
        ids = get_node_ids_by_type(graph.finalize(), "KSamplerAdvanced")
        for id in ids:
            real_id = id.replace(f"{unique_id}.0.0.", "")
            node = graph.lookup_node(real_id)
            seed = node.get_input("noise_seed")
            node.set_input("noise_seed", seed + d["index"] + 1)

        # update the store in the new_combine (this node)
        new_combine = graph.lookup_node("Recurse")
        new_combine.set_input("store", s)

        ui_values = {
            "input_label_values": {
                "images": len(images) if images is not None else 0,
                "masks": len(masks) if masks is not None else 0,
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
        }

        log(f"Finished chunk {d["index"] + 1} of {c["chunk_count"]}")

        return {
            "ui": {"values": [ui_values]},
            "result": (
                new_combine.out(0),
                new_combine.out(1),
                new_combine.out(2),
                new_combine.out(3),
            ),
            "expand": graph.finalize(),
        }
