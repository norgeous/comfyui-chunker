import torch
import math
from comfy_execution.graph_utils import GraphBuilder
from ..lib.utils import (
    log,
    mask_to_image,
    image_to_mask,
)
from ..lib.debug_overlay import create_preview_video
from ..lib.repeat_nodes import comfyui_repeat_nodes, get_node_ids_by_type
from ..lib.av.loader import awesome_loader, quick_combine, save_video, save_audio
from ..lib.av.load_audio import concat_audios
from ..lib.format_utils import format_audio, format_fps

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
        if images is None and masks is None:
            raise Exception("Please provide images OR masks")

        d = chunker_data
        c = d["chunker_config"]
        s = store if store is not None else {
            "image_chunks": [],
            "mask_chunks": [],
            "audio_chunks": [],
            "preview_chunks": [],
        }

        # figure out if we have completed all chunks
        is_done = d["index"] + 1 >= c["chunk_count"]

        # save new image chunk to a new file
        if images is not None:
            images_full_path = save_video(images, d["fps"], "video/chunker/tmp/chunk/image_chunk", audio)[0]
            s["image_chunks"].append(images_full_path)

        # save new mask chunk to a new file
        if masks is not None:
            masks_full_path = save_video(mask_to_image(masks), d["fps"], "video/chunker/tmp/chunk/mask_chunk")[0]
            s["mask_chunks"].append(masks_full_path)

        # save new audio chunk to a new file
        if audio is not None:
            audio_full_path = save_audio(audio, "video/chunker/tmp/chunk/audio_chunk")[0]
            print(audio_full_path)
            # s["audio_chunks"].append(audio)
            s["audio_chunks"].append(audio_full_path)

        # create preview from inputs
        preview = create_preview_video(images, masks, show_debug, d, c)

        # save new preview chunk to a new file
        preview_full_path = save_video(preview, d["fps"], "video/chunker/tmp/chunk/preview_chunk", audio)[0]
        s["preview_chunks"].append(preview_full_path)

        #print("these should have audio", s["image_chunks"], s["preview_chunks"])

        # combine all preview chunks to a new file, excluding the overlaps
        filename_prefix = "video/chunker/tmp/chunks/preview_chunks" if not is_done else "video/chunker/tmp/chunks/preview_complete"
        all_preview_frontend_data = quick_combine(s["preview_chunks"], c["chunk_overlap"], select_overlaps_from, filename_prefix)[1]

        # if no more chunks needed, return early
        if is_done:
            # load all image chunks as tensor, excluding the overlaps
            out_images_torch = None
            if len(s["image_chunks"]) > 0:
                log("[debug] Combine -> combine all images...", end="")
                all_images_video_path = quick_combine(s["image_chunks"], c["chunk_overlap"], select_overlaps_from, "video/chunker/images")[0]
                print("done")
                log("[debug] Combine -> load all images as tensor...", end="")
                out_images_torch = awesome_loader(all_images_video_path)[0]
                print("done")

            # load all mask chunks as tensor, excluding the overlaps
            out_masks_torch = None
            if len(s["mask_chunks"]) > 0:
                log("[debug] Combine -> combine all masks...", end="")
                all_masks_video_path = quick_combine(s["mask_chunks"], c["chunk_overlap"], select_overlaps_from, "video/chunker/masks")[0]
                print("done")
                log("[debug] Combine -> load all masks as tensor...", end="")
                out_masks_torch = awesome_loader(all_masks_video_path)[0]
                print("done")

            # load all audio chunks as tensor
            out_audio_dict = None
            if len(s["audio_chunks"]) > 0:
                out_audio_dict = concat_audios(s["audio_chunks"])
                # load_audio(d["audio"]) if d["audio"] is not None else None,

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
                    image_to_mask(out_masks_torch),
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
            "images_last_chunk_path": s["image_chunks"][-1] if len(s["image_chunks"]) > 0 else None, # filename of last image chunk saved
            "masks_last_chunk_path": s["mask_chunks"][-1] if len(s["mask_chunks"]) > 0 else None, # filename of last mask chunk saved
            "audio_last_chunk_path": s["audio_chunks"][-1] if len(s["audio_chunks"]) > 0 else None, # filename of last audio chunk saved
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

        # increment seeds in cloned mmaudio
        #ids = get_node_ids_by_type(graph.finalize(), "MMAudioSampler")
        #for id in ids:
        #    real_id = id.replace(f"{unique_id}.0.0.", "")
        #    node = graph.lookup_node(real_id)
        #    seed = node.get_input("seed")
        #    node.set_input("seed", seed + d["index"] + 1)

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
