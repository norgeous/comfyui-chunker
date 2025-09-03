import torch
import math
from comfy_execution.graph_utils import GraphBuilder
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
from .utils import log, panelImage, panelMask, overlay_debug
from .repeatNodes import comfyuiRepeatNodes, getNodeIdsByType
from .saveVideo import save_video


class ChunkerConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["Wan","None"], {"tooltip": "TODO"}),
                "chunk_length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4, "tooltip": "Count of images in each chunk"}),
                "chunk_overlap": ("INT", {"default": 4, "min": 0, "max": 4096, "step": 1, "tooltip": "Count of images to overlap between chunks"}),
                "total_length": ("INT", {"default": 500, "min": 1, "max": 100000, "step": 1, "tooltip": "Minimum count of images in the final output"}),
            },
        }

    RETURN_TYPES = ("CHUNKER_CONFIG", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("chunker_config", "chunk_length","chunk_overlap","total_length","chunk_count")
    OUTPUT_TOOLTIPS = (
        "Includes all settings",
        "Count of images in each chunk",
        "Count of images to overlap between each chunk",
        "Total length of output images",
        "Count of chunks",
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
        chunk_count = math.ceil((total_length - chunk_overlap) / (chunk_length - chunk_overlap))

        if mode == "Wan":
            # we want to avoid the situation where the last chunk of images is not a valid length for Wan
            # adjust total_length, so that the final chunk matches 4n+1
            previous_chunks_total = (chunk_length * (chunk_count - 1)) - (chunk_overlap * (chunk_count - 1))
            final_chunk_length = total_length - previous_chunks_total
            adjusted_final_chunk_length = (round(final_chunk_length / 4) * 4) + 1 # force 4n+1
            total_length = previous_chunks_total + adjusted_final_chunk_length

        chunker_config = {
            "mode": mode,
            "chunk_length": chunk_length,
            "chunk_overlap": chunk_overlap,
            "total_length": total_length,
            "chunk_count": chunk_count,
        }

        ui_values = {
            "output_label_values": {
                "chunk_length": chunk_length,
                "chunk_overlap": chunk_overlap,
                "total_length": total_length,
                "chunk_count": chunk_count,
            },
        }

        return {
            "ui": {"values": [ui_values]},
            "result": (
                chunker_config,
                chunk_length,
                chunk_overlap,
                total_length,
                chunk_count,
            ),
        }


class ChunkerResequencer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunker_config": ("CHUNKER_CONFIG", {"tooltip": "TODO"}),
                "start_image_count": ("INT", {"tooltip": "TODO"}),
                "use_end_image": (["every_chunk", "final_chunk_only", "never"], {"tooltip": "TODO"}),
                #"sequence": ("STRING", {"default": "", "multiline": True, "tooltip": "TODO"}),
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
        start_image_count,
        use_end_image,
        images=None,
        masks=None,
    ):
        c = chunker_config
        w = images.shape[2] if images is not None else 512
        h = images.shape[1] if images is not None else 512

        grey_panel  = panelImage(w, h, 127, 127, 127)
        black_panel = panelMask(w, h, 0)
        white_panel = panelMask(w, h, 255)

        out_images = []
        out_masks = []

        # for every frame in sequence, select a frame
        given_count = len(images) if images is not None else 0
        given_count_masks = len(masks) if masks is not None else 0
        used_count = 0
        for i in range(c["total_length"]):
            if given_count > 0:
                if (
                    i < start_image_count
                    or
                    use_end_image == "every_chunk" and i > c["chunk_overlap"] and ((i+1 - c["chunk_overlap"]) / (c["chunk_length"] - c["chunk_overlap"])) % 1 == 0
                    or
                    use_end_image in ["every_chunk","final_chunk_only"] and i == c["total_length"] - 1
                ):
                    next = min(used_count, given_count-1)
                    out_images.extend([images[next:next + 1]])
                    out_masks.extend([masks[next:next + 1]] if given_count_masks > next else [black_panel])
                    used_count += 1
                    continue

            out_images.extend([grey_panel])
            out_masks.extend([white_panel])

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
                "chunker_config": ("CHUNKER_CONFIG", {"tooltip": "tbd"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Images to be chunked"}),
                "masks": ("MASK", {"tooltip": "Masks to be chunked"}),
                "store": ("*",), # hidden by js
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("CHUNKER_DATA", "IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("chunker_data", "images", "masks", "width", "height", "chunk_length", "index")
    OUTPUT_TOOLTIPS = (
        "Connect \"chunker_data\" to the \"ChunkerCombine\" node",
        "Chunk of images",
        "Chunk of masks",
        "Width of images",
        "Height of images",
        "Length of current chunk",
        "The current itteration index, ie; 0, 1, 2, ...",
    )
    FUNCTION = "main"
    CATEGORY = "Chunker"
    DESCRIPTION = "Chunker"

    def main(
        self,
        chunker_config,
        images=None,
        masks=None,
        store=None,
        unique_id=None,
    ):
        if images is None and masks is None:
            raise Exception("Please provide images OR masks")

        c = chunker_config
        s = {
            "index": 0,
            "images_overlap": None,
        } if store is None else store

        w = images.shape[2] if images is not None else 512
        h = images.shape[1] if images is not None else 512

        log(f"Starting chunk {s["index"] + 1} of {c["chunk_count"]}...")

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

        this_chunk_length = max(len(out_images_torch), len(out_masks_torch))

        if c["mode"] == "Wan":
            this_chunk_length = (round(this_chunk_length / 4) * 4) + 1 # force 4n+1

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
                "width": w,
                "height": h,
                "chunk_length": this_chunk_length,
                "index": s["index"],
            },
        }

        return {
            "ui": {"values": [ui_values]},
            "result": (
                chunker_data,
                out_images_torch, # just this chunk
                out_masks_torch, # just this chunk
                w,
                h,
                this_chunk_length,
                s["index"],
            ),
        }


class ChunkerCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunker_data": ("CHUNKER_DATA", {"tooltip": "Connect chunker_data from Chunker node to here"}),
                "preview_fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "step": 1.0, "tooltip": "The FPS of the preview video"}),
                "show_debug": ("BOOLEAN", {"default": True, "tooltip": "Show debug overlay in preview"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Processed chunk of images"}),
                "masks": ("MASK", {"tooltip": "Processed chunk of masks"}),
                "store": ("*",), # hidden by js
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
        chunker_data,
        preview_fps,
        show_debug,
        images=None,
        masks=None,
        dynprompt=None,
        unique_id=None,
        store=None,
        *args,
        **kwargs
    ):
        log(args, kwargs)
        log("store:",store)
        d = chunker_data
        c = d["chunker_config"]
        s = {
            "images_previous": None,
            "masks_previous": None,
            "preview_previous": None,
        } if store is None else store

        log(f"Finished chunk {d["index"] + 1} of {c["chunk_count"]}!")

        # combine all chunks so far
        out_images = []
        if s["images_previous"] is not None: out_images.extend([s["images_previous"]])
        out_images.extend([images])
        out_images_torch = torch.cat(out_images)

        is_done = len(out_images_torch) >= c["total_length"]
        log(f"{is_done} = {len(out_images_torch)} >= {c['total_length']}")

        # make preview video for new images
        preview_video = []
        if s["preview_previous"] is not None: preview_video.extend([s["preview_previous"]])
        previous_count = len(s["preview_previous"]) if s["preview_previous"] is not None else 0
        preview_video_chunk = overlay_debug(images, previous_count, d["index"], c["chunk_count"], c["chunk_length"], c["chunk_overlap"], c["total_length"]) if show_debug else out_images_torch
        preview_video.extend([preview_video_chunk])
        preview_video_torch = torch.cat(preview_video)

        # save preview video
        filename_prefix = "video/chunker/tmp/tmp" if not is_done else "video/chunker/tmp/complete"
        video_path = save_video(preview_video_torch, preview_fps, filename_prefix)

        # if no more chunks needed return early
        if is_done:
            ui_values = {
                "input_label_values": {
                    "images": len(images) if images is not None else None,
                    "masks": len(masks) if masks is not None else None,
                },
                "output_label_values": {
                    "images": len(out_images_torch),
                    #"masks": len(out_masks_torch),
                },
                # "image_count": image_count,
                "index": d["index"],
                "chunk_count": c["chunk_count"],
                "video_path": video_path,
            }
            return {
                "ui": {"values": [ui_values]},
                "result":(
                    out_images_torch,
                    #out_masks_torch,
                )
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
            "preview_previous": preview_video_torch[:-c["chunk_overlap"]] if c["chunk_overlap"] > 0 else preview_video_torch,
        })
        print(new_combine)

        ui_values = {
            "input_label_values": {
                "images": len(images) if images is not None else None,
                "masks": len(masks) if masks is not None else None,
            },
            "output_label_values": {
                "images": None,
                "masks": None,
            },
            "index": d["index"],
            "chunk_count": c["chunk_count"],
            "video_path": video_path,
        }

        return {
            "ui": {"values": [ui_values]},
            "result": (new_combine.out(0),),
            "expand": graph.finalize(),
        }
