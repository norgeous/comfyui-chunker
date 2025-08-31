import torch
import math
from datetime import datetime
from comfy_execution.graph_utils import GraphBuilder
from comfy_extras.nodes_video import CreateVideo, SaveVideo
from comfy.utils import ProgressBar
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
from .utils import log, panelImage, panelMask, slice, len2, resizeImage, resizeMask, overlay_debug
from .repeatNodes import comfyuiRepeatNodes, getNodeIdsByType


class ChunkerConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["wan"], {"tooltip": "TODO"}),
                "chunk_length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4, "tooltip": "Count of images in each chunk"}),
                "chunk_overlap": ("INT", {"default": 4, "min": 0, "max": 4096, "step": 1, "tooltip": "Count of images to overlap between chunks"}),
                "total_length": ("INT", {"default": 500, "min": 1, "max": 100000, "step": 1, "tooltip": "Minimum count of images in the final output"}),
            },
        }

    RETURN_TYPES = ("CHUNKER_CONFIG", "INT")
    RETURN_NAMES = ("chunker_config", "chunk_count")
    OUTPUT_TOOLTIPS = (
        "includes all settings",
    )
    FUNCTION = "main"
    CATEGORY = "Chunker"
    DESCRIPTION = "Chunker config"

    def main(
        self,
        mode,
        chunk_length,
        chunk_overlap,
        total_length,
    ):
        loop_count = math.ceil((total_length - chunk_overlap) / (chunk_length - chunk_overlap))

        chunker_config = {
            "mode": mode,
            "chunk_length": chunk_length,
            "chunk_overlap": chunk_overlap,
            "total_length": total_length,
            "loop_count": loop_count,
        },

        ui_values = {   
            "output_label_values": {
                "chunk_count": loop_count,
            },
        }

        return {
            "ui": {"values": [ui_values]},
            "result": (
                chunker_config,
                loop_count,
            ),
        }


class ChunkerSequencer:#ChunkerRemix
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunker_config": ("CHUNKER_CONFIG", {"tooltip": "TODO"}),
                "sequence": ("STRING", {"default": "", "multiline": True, "tooltip": "TODO"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "None, single image or batch of images to be resequenced"}),
                "masks": ("MASK", {"tooltip": "None, single mask or batch of masks to be used in resequenced"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    OUTPUT_TOOLTIPS = (
        "remixed sequence of images",
        "remixed sequence of masks",
    )
    FUNCTION = "main"
    CATEGORY = "Chunker"
    DESCRIPTION = "Remix or generate image and mask sequences"

    def main(
        self,
        chunker_config,
        sequence,
        images=None,
        masks=None,
    ):
        c = chunker_config
        w = images.shape[2] if images is not None else 512
        h = images.shape[1] if images is not None else 512

        grey_panel  = panelImage(w, h, 127, 127, 127)
        #black_panel = panelMask(w, h, 0)
        white_panel = panelMask(w, h, 255)

        out_images = []
        out_images.extend([grey_panel] * c["total_length"])

        out_masks = []
        out_masks.extend([white_panel] * c["total_length"])

        out_images_torch = torch.cat(out_images)
        out_masks_torch = torch.cat(out_masks)

        ui_values = {
            "input_label_values": {
                "images": len(images) if images is not None else 0,
                "masks": len(masks) if masks is not None else 0,
            },
            "output_label_values": {
                "images": len(out_images_torch),
                "masks": len(out_masks_torch),
            },
        }

        return {
            "ui": {"values": [ui_values]},
            "result": (
                out_images_torch,
                out_masks_torch,
            ),
        }


class Chunker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "store": ("INT",), # hidden by js
                "chunker_config": ("CHUNKER_CONFIG", {"tooltip": "tbd"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Images to be chunked"}),
                "masks": ("MASK", {"tooltip": "Masks to be chunked"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("CHUNKER_DATA", "IMAGE", "MASK", "INT")
    RETURN_NAMES = ("chunker_data", "images", "masks", "index")
    OUTPUT_TOOLTIPS = (
        "Connect \"chunker_data\" to the \"ChunkerCombine\" node",
        "Chunk of images",
        "Chunk of masks",
        "The current itteration index, ie; 0, 1, 2, ...",
    )
    FUNCTION = "main"
    CATEGORY = "Chunker"
    DESCRIPTION = "Chunker"

    def main(
        self,
        store,
        chunker_config,
        images=None,
        masks=None,
        unique_id=None,
    ):
        if images is None and masks is None:
            raise Exception("Please provide images OR masks")

        c = chunker_config
        s = {
            "index": 0,
            "images_overlap": None,
        } if type(store) is int else store

        w = images.shape[2] if images is not None else 512
        h = images.shape[1] if images is not None else 512

        log(f"Starting chunk {s["index"] + 1} of {c["loop_count"]}...")

        black_panel = panelMask(w, h, 0)

        out_images = []
        out_masks = []

        adjusted_chunk_overlap = 0 if s["index"] == 0 else c["chunk_overlap"]

        # add the images_overlap if they exist
        if s["images_overlap"] is not None:
            out_images.extend([s["images_overlap"]])

            # add as many black masks as overlap length
            out_masks.extend([black_panel] * len(s["images_overlap"]))

        # cut chunk from images and masks and add them
        start = s["index"] * (c["chunk_length"] - c["chunk_overlap"]) + adjusted_chunk_overlap
        end = start + c["chunk_length"] - adjusted_chunk_overlap
        out_images.extend([images[start:end]])
        out_masks.extend([masks[start:end]])

        # convert to tensor
        out_images_torch = torch.cat(out_images)
        out_masks_torch = torch.cat(out_masks)

        chunker_data = {
            "start_node_id": unique_id,
            "index": s["index"],
            "chunker_config": c,
        }

        ui_values = {
            "input_label_values": {
                "images": len(images),
                "masks": len(masks),
            },
            "output_label_values": {
                "images": len(out_images_torch),
                "masks": len(out_masks_torch),
                "index": s["index"],
            },
        }

        return {
            "ui": {"values": [ui_values]},
            "result": (
                chunker_data,
                out_images_torch, # just this chunk
                out_masks_torch, # just this chunk
                s["index"],
            ),
        }


class ChunkerCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "store": ("INT",), # hidden by js
                "chunker_data": ("CHUNKER_DATA", {"tooltip": "Connect chunker_data from Chunker node to here"}),
                "preview_fps": ("FLOAT", {"default": 16, "min": 1, "max": 120, "step": 1, "tooltip": "The FPS of the preview video"}),
                "show_debug": ("BOOLEAN", {"default": True, "tooltip": "Show debug overlay in preview"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Processed chunk of images"}),
                "masks": ("MASK", {"tooltip": "Processed chunk of masks"}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    OUTPUT_TOOLTIPS = (
        "Combined images from all chunks",
        "Combined masks from all chunks",
    )
    FUNCTION = "main"
    CATEGORY = "Chunker"
    DESCRIPTION = "ChunkerCombine"
    OUTPUT_NODE = True

    def main(
        self,
        store,
        chunker_data,
        images,
        masks,
        show_debug,
        preview_fps,
        dynprompt=None,
        unique_id=None,
    ):
        d = chunker_data
        c = d["chunker_config"]
        s = {
            "images_previous": None,
            "masks_previous": None,
        } if type(store) is int else store

        log(f"Finished chunk {d["index"] + 1} of {c["loop_count"]}!")

        # combine all chunks so far
        out_images = []
        if s["images_previous"] is not None: out_images.extend([s["images_previous"]])
        out_images.extend([images])
        out_images_torch = torch.cat(out_images)

        is_done = len(out_images_torch) >= c["total_length"]
        log(is_done,'=',len(out_images_torch) ,'>=', c["total_length"])

        # make preview video
        preview_video_torch = overlay_debug(out_images_torch, c["chunk_length"], c["chunk_overlap"]) if show_debug else out_images_torch

        # save preview video
        create_video_node = CreateVideo()
        video, = create_video_node.create_video(preview_video_torch, preview_fps)
        save_video_node = SaveVideo()
        save_to = "video/chunker/tmp/tmp" if not is_done else 'video/chunker/tmp/complete'
        save_result = save_video_node.save_video(video, save_to, "auto", "auto")
        video_path = save_result["ui"]["images"][0]

        # if no more chunks needed return early
        if is_done:
            ui_values = {
                "output_label_values": {
                    "images": len(out_images_torch),
                },
                # "image_count": image_count,
                "index": d["index"],
                "loop_count": c["loop_count"],
                "video_path": video_path,
            }
            return {
                "ui": {"values": [ui_values]},
                "result":(out_images_torch,)
            }


        # clone all the nodes between Chunker and ChunkerCombine
        graph = GraphBuilder()
        comfyuiRepeatNodes(dynprompt, graph, unique_id, d["start_node_id"])

        # update the store in the new_chunker
        new_chunker = graph.lookup_node(d["start_node_id"])
        new_chunker.set_input("store", {
            "index": d["index"] + 1,
            "images_overlap": images[-c["chunk_overlap"]:] if c["chunk_overlap"] > 0 else None
        })

        # increment seeds in cloned KSamplers, to prevent same motion in each chunk (for Wan)
        ids = getNodeIdsByType(graph.finalize(), "KSampler")
        for id in ids:
            real_id = id.replace(f"{unique_id}.0.0.", "")
            node = graph.lookup_node(real_id)
            seed = node.get_input("seed")
            node.set_input("seed", seed + d["index"] + 1)

        # update the store in the new_combine (this node)
        new_combine = graph.lookup_node("Recurse")
        new_combine.set_input("store", {
            "images_previous": out_images_torch[:-c["chunk_overlap"]] if c["chunk_overlap"] > 0 else out_images_torch,
        })

        ui_values = {
            "output_label_values": {
                "images": None,
            },
            "index": d["index"],
            "loop_count": c["loop_count"],
            "video_path": video_path,
        }

        return {
            "ui": {"values": [ui_values]},
            "result": (new_combine.out(0),),
            "expand": graph.finalize(),
        }
