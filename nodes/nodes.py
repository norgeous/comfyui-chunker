import torch
import math
from comfy_execution.graph_utils import GraphBuilder
from comfy_api.input_impl import VideoFromFile
from .utils import log, panelImage, panelMask, mask_to_image, overlay_debug
from .repeatNodes import comfyuiRepeatNodes, getNodeIdsByType
from .video import save_video, load_video_images_exclude_overlap


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
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    DESCRIPTION = "Chunker config"

    def execute(
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
                "start_image": ("BOOLEAN", {"tooltip": "TODO"}),
                "image_count_every_chunk": ("INT", {"tooltip": "Count of images to be used in each chunk"}),
                "end_image_every_chunk": ("BOOLEAN", {"tooltip": "TODO"}),
                "end_image": ("BOOLEAN", {"tooltip": "TODO"}),
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
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    DESCRIPTION = "Remix or generate image and mask sequences"

    def execute(
        self,
        chunker_config,
        start_image,
        image_count_every_chunk,
        end_image_every_chunk,
        end_image,
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
        if images is None:
            out_images.extend([grey_panel] * c["total_length"])
            out_masks.extend([white_panel] * c["total_length"])
        else:
            next_image = 0
            next_mask = 0
            middle_images_positions = []
            delta = c["chunk_length"] / (image_count_every_chunk + 1)
            for i in range(1, image_count_every_chunk + 1):
                middle_images_positions.append(round(i * delta))
            for i in range(c["total_length"]):
                chunk_position_fraction = max(0, i+1 - c["chunk_overlap"]) / (c["chunk_length"] - c["chunk_overlap"]) % 1
                is_end = i > c["chunk_overlap"] and chunk_position_fraction == 0

                if (
                    (start_image and i == 0)
                    or
                    (image_count_every_chunk > 0 and i % c["chunk_length"] in middle_images_positions)
                    or
                    (end_image_every_chunk and is_end)
                    or
                    (end_image and i == c["total_length"] - 1)
                ):
                    # then add the next image at this index
                    out_images.extend([images[next_image:next_image+1]])
                    next_image = (next_image + 1) % len(images)

                    # and add a mask at the same index
                    if masks is not None:
                        out_masks.extend([masks[next_mask:next_mask+1]])
                        next_mask = (next_mask + 1) % len(masks)
                    else:
                        out_masks.extend([black_panel])

                    continue

                # otherwise this index is "blank"
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
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    DESCRIPTION = "Chunker"

    def execute(
        self,
        chunker_config,
        images=None,
        masks=None,
        unique_id=None,
        store={
            "index": 0,
            "images_overlap": None,
            "masks_overlap": None,
        }
    ):
        if images is None and masks is None:
            raise Exception("Please provide images OR masks")

        c = chunker_config
        s = store

        w = images.shape[2] if images is not None else 512
        h = images.shape[1] if images is not None else 512

        log(f"Starting chunk {s["index"] + 1} of {c["chunk_count"]}...")

        black_panel = panelMask(w, h, 0)

        out_images = []
        out_masks = []

        adjusted_chunk_overlap = 0 if s["index"] == 0 else c["chunk_overlap"]
        images_overlap_count = 0 if s["images_overlap"] is None else len(s["images_overlap"])
        masks_overlap_count = 0 if s["masks_overlap"] is None else len(s["masks_overlap"])

        # add images_overlap if it exists
        if s["images_overlap"] is not None: out_images.extend([s["images_overlap"]])

        # add masks_overlap if it exists
        if s["masks_overlap"] is not None:
            out_masks.extend([s["masks_overlap"]])
            #out_masks.extend(s["masks_overlap"])
        else:
            # add as many black masks as images in overlap
            if s["images_overlap"] is not None:
                out_masks.extend([black_panel] * len(s["images_overlap"]))

        # cut chunk from images and masks and add them
        #start = s["index"] * (c["chunk_length"] - c["chunk_overlap"]) + adjusted_chunk_overlap
        #istart = s["index"] * (c["chunk_length"] - c["chunk_overlap"]) + images_overlap_count
        #mstart = s["index"] * (c["chunk_length"] - c["chunk_overlap"]) + masks_overlap_count
        #iend = istart + c["chunk_length"] - adjusted_chunk_overlap
        #mend = mstart + c["chunk_length"] - adjusted_chunk_overlap


        start = s["index"] * (c["chunk_length"] - c["chunk_overlap"])
        end = start + c["chunk_length"]

        if images is not None: out_images.extend([images[start + images_overlap_count:end]])
        if masks is not None: out_masks.extend([masks[start + masks_overlap_count:end]])

        # convert to tensor
        #log(out_images)
        #log(out_masks)

        #log("M", masks, start, end)
        #if masks is not None: log("M.s", masks.shape)

        #log("MO", s["masks_overlap"])
        #if s["masks_overlap"] is not None: log("MO.s", s["masks_overlap"].shape)

        out_images_torch = torch.cat(out_images) if len(out_images) > 0 else None
        out_masks_torch = torch.cat(out_masks) if len(out_masks) > 0 else None # that?

        this_chunk_length = max(
            len(out_images_torch) if out_images_torch is not None else 0,
            len(out_masks_torch) if out_masks_torch is not None else 0,
        )

        chunker_data = {
            "start_node_id": unique_id,
            "index": s["index"],
            "chunker_config": c,
        }

        ui_values = {
            "input_label_values": {
                "images": len(images) if images is not None else 0,
                "masks": len(masks) if masks is not None else 0,
            },
            "output_label_values": {
                "images": len(out_images_torch) if out_images_torch is not None else 0,
                "masks": len(out_masks_torch) if out_masks_torch is not None else 0,
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
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    DESCRIPTION = "ChunkerCombine"
    OUTPUT_NODE = True

    def execute(
        self,
        chunker_data,
        preview_fps,
        show_debug,
        images=None,
        masks=None,
        dynprompt=None,
        unique_id=None,
        store={
            "images_previous": None,
            "masks_previous": None,
            #"preview_previous": None,
            "preview_previous_video_path": None,
        },
    ):
        if images is None and masks is None:
            raise Exception("Please provide images OR masks")

        d = chunker_data
        c = d["chunker_config"]
        s = store

        # combine previous chunks with new chunks
        log("torch.cat images")
        out_images = []
        if s["images_previous"] is not None: out_images.extend([s["images_previous"]])
        if images is not None: out_images.extend([images])
        out_images_torch = torch.cat(out_images) if len(out_images) > 0 else None

        out_masks = []
        log("torch.cat masks")
        if s["masks_previous"] is not None: out_masks.extend([s["masks_previous"]])
        if masks is not None: out_masks.extend([masks])
        out_masks_torch = torch.cat(out_masks) if len(out_masks) > 0 else None

        is_done = d["index"] + 1 >= c["chunk_count"]

        # make preview video for new images and masks
        preview_video_torch = None
        video_path = None

        # preview_previous = VideoFromFile(s["preview_previous_video_path"]).get_components().images if s["preview_previous_video_path"] is not None else None
        # preview_previous = preview_previous[:-c["chunk_overlap"]] if c["chunk_overlap"] > 0 and preview_previous is not None else None
        # log("loaded video images", preview_previous)

        preview_previous = load_video_images_exclude_overlap(s["preview_previous_video_path"], c["chunk_overlap"])

        # add previous preview
        preview_video = []
        #if s["preview_previous"] is not None: preview_video.extend([s["preview_previous"]])
        #previous_count = len(s["preview_previous"]) if s["preview_previous"] is not None else 0
        if preview_previous is not None: preview_video.extend([preview_previous])
        previous_count = len(preview_previous) if preview_previous is not None else 0

        if masks is not None and images is None:
            # convert mask to image
            imasks = mask_to_image(masks)
            preview_video_chunk = overlay_debug(imasks, previous_count, d["index"], c["chunk_count"], c["chunk_length"], c["chunk_overlap"], c["total_length"]) if show_debug else imasks
            preview_video.extend([preview_video_chunk])
            log("torch.cat preview")
            preview_video_torch = torch.cat(preview_video)
        if images is not None and masks is None:
            preview_video_chunk = overlay_debug(images, previous_count, d["index"], c["chunk_count"], c["chunk_length"], c["chunk_overlap"], c["total_length"]) if show_debug else images
            preview_video.extend([preview_video_chunk])
            log("torch.cat preview")
            preview_video_torch = torch.cat(preview_video)

        # save preview video
        if preview_video_torch is not None:
            filename_prefix = "video/chunker/tmp/tmp" if not is_done else "video/chunker/tmp/complete"
            video_path, full_path = save_video(preview_video_torch, preview_fps, filename_prefix)

        log("video_path", video_path)

        # if no more chunks needed return early
        if is_done:
            ui_values = {
                "input_label_values": {
                    "images": len(images) if images is not None else 0,
                    "masks": len(masks) if masks is not None else 0,
                },
                "output_label_values": {
                    "images": len(out_images_torch) if out_images_torch is not None else 0,
                    "masks": len(out_masks_torch) if out_masks_torch is not None else 0,
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
                    out_masks_torch,
                )
            }

        # clone all the nodes between Chunker and ChunkerCombine
        graph = GraphBuilder()
        comfyuiRepeatNodes(dynprompt, graph, unique_id, d["start_node_id"])

        # update the store in the new_chunker
        new_chunker = graph.lookup_node(d["start_node_id"])
        new_chunker.set_input("store", {
            "index": d["index"] + 1,
            "images_overlap": images[-c["chunk_overlap"]:] if c["chunk_overlap"] > 0 and images is not None else None,
            "masks_overlap": masks[-c["chunk_overlap"]:] if c["chunk_overlap"] > 0 and masks is not None else None,
        })

        # increment seeds in cloned KSamplers, to prevent same motion in each chunk (for Wan)
        ids = getNodeIdsByType(graph.finalize(), "KSampler")
        for id in ids:
            real_id = id.replace(f"{unique_id}.0.0.", "")
            node = graph.lookup_node(real_id)
            seed = node.get_input("seed")
            node.set_input("seed", seed + d["index"] + 1)

        log("big slice operations next")

        # update the store in the new_combine (this node)
        new_combine = graph.lookup_node("Recurse")
        new_combine.set_input("store", {
            "images_previous": out_images_torch[:-c["chunk_overlap"]] if c["chunk_overlap"] > 0 and out_images_torch is not None else out_images_torch,
            "masks_previous": out_masks_torch[:-c["chunk_overlap"]] if c["chunk_overlap"] > 0 and out_masks_torch is not None else out_masks_torch,
            # "preview_previous_video_path": f"/home/user/ComfyUI/output/{video_path['subfolder']}/{video_path['filename']}", # TODO path.join
            "preview_previous_video_path": full_path
        })

        ui_values = {
            "input_label_values": {
                "images": len(images) if images is not None else 0,
                "masks": len(masks) if masks is not None else 0,
            },
            "output_label_values": {
                "images": None,
                "masks": None,
            },
            "index": d["index"],
            "chunk_count": c["chunk_count"],
            "video_path": video_path,
        }

        log(f"Finished chunk {d["index"] + 1} of {c["chunk_count"]}!")

        return {
            "ui": {"values": [ui_values]},
            "result": (
                new_combine.out(0),
                new_combine.out(1),
            ),
            "expand": graph.finalize(),
        }
