import os
import folder_paths
import torch
import math
#from comfy_api.latest import ComfyExtension, io
from comfy_execution.graph_utils import GraphBuilder
from .utils import log, panelImage, panelMask, mask_to_image, image_to_mask, resize_image, resize_mask, create_preview_video, get_input_filenames
from .repeatNodes import comfyuiRepeatNodes, getNodeIdsByType
from .video import save_video, ffmpeg_first_frame
from .loader import awesome_loader

# Add custom API routes, using router
from aiohttp import web
from server import PromptServer
from urllib.parse import unquote

# register /api/chunker/get-first-frame
@PromptServer.instance.routes.get("/chunker/get-first-frame")
async def get_hello(request):
    if "filename" in request.query:
        input_dir = folder_paths.get_input_directory()
        filename = os.path.join(input_dir, unquote(request.query["filename"]))
        if not os.path.isfile(filename):
            return web.HTTPBadRequest()
        image_path_data = ffmpeg_first_frame(filename)
        return web.json_response(image_path_data)
    else:
        return web.HTTPBadRequest()


class Chunker:
    @classmethod
    def INPUT_TYPES(cls):
        files = ["None", *sorted(get_input_filenames())]
        return {
            "required": {
                "mode": (["None", "Wan"], {"tooltip": "TODO"}),
                "chunk_length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4, "tooltip": "Count of images in each chunk"}),
                "chunk_overlap": ("INT", {"default": 4, "min": 0, "max": 4096, "step": 1, "tooltip": "Count of images to overlap between chunks"}),
                "total_length": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "Minimum count of images in the final output. 0 to use the images length"}),
                "images": (files, {"default": "None", "tooltip": "Images to be chunked"}),
                "masks": (files, {"default": "None", "tooltip": "Masks to be chunked"}),
            },
            "optional": {
                "store": ("*",), # hidden by js
                "image": (files,),
                "image_paint": (files,),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, mode, chunk_length, chunk_overlap, total_length, images, masks, image, image_paint):
        # YOLO, anything goes!
        return True

    RETURN_TYPES = ("CHUNKER_DATA", "IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("chunker_data", "images", "masks", "width", "height", "index", "chunk_length","chunk_overlap","total_length","chunk_count")
    OUTPUT_TOOLTIPS = (
        "Connect \"chunker_data\" to the \"ChunkerCombine\" node",
        "Chunk of images",
        "Chunk of masks",
        "Width of images",
        "Height of images",
        "The current itteration index, ie; 0, 1, 2, ...",
        "Count of images in each chunk",
        "Count of images to overlap between each chunk",
        "Total length of output images",
        "Count of chunks",
    )
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    DESCRIPTION = "Chunker"

    def execute(
        self,
        mode,
        chunk_length,
        chunk_overlap,
        total_length,
        images="None",
        masks="None",
        image="None",
        image_paint="None",
        store=None,
        unique_id=None,
    ):
        s = store if store is not None else {
            "index": 0,
            "images_last_chunk_path": None,
            "masks_last_chunk_path": None,
        }

        #if mode == "Wan":
        #    # we want to avoid the situation where the last chunk of images is not a valid length for Wan
        #    # adjust total_length, so that the final chunk matches 4n+1
        #    previous_chunks_total = (chunk_length * (chunk_count - 1)) - (chunk_overlap * (chunk_count - 1))
        #    final_chunk_length = total_length - previous_chunks_total
        #    adjusted_final_chunk_length = (round(final_chunk_length / 4) * 4) + 1 # force 4n+1
        #    total_length = previous_chunks_total + adjusted_final_chunk_length

        start = s["index"] * (chunk_length - chunk_overlap)
        end = start + chunk_length

        if images == "None": images = None
        if masks == "None": masks = None
        if image == "None": image = None
        if image_paint == "None": image_paint = None

        w = 512
        h = 512
        fps = 30

        # get the mask from the mask editor for first chunk only
        mask_maskeditor = None
        if image is not None and s["index"] == 0:
            mask_editor_filename = image.replace("clipspace/", "").replace(" [input]", "")
            path_full = os.path.join(folder_paths.get_input_directory(), 'clipspace', mask_editor_filename)
            mask_maskeditor = awesome_loader(path_full)[0]

        # load the images overlap from file
        images_overlap = None
        if s["images_last_chunk_path"] is not None:
            images_overlap = awesome_loader(s["images_last_chunk_path"], start=-chunk_overlap)[0]

        # load the masks overlap from file
        masks_overlap = None
        if s["masks_last_chunk_path"] is not None:
            masks_overlap = awesome_loader(s["masks_last_chunk_path"], start=-chunk_overlap)[0]

        count_maskeditor = 1 if mask_maskeditor is not None else 0
        count_images_overlap = len(images_overlap) if images_overlap is not None else 0
        count_masks_overlap = len(masks_overlap) if masks_overlap is not None else 0

        # get images chunk from input file and construct the output
        out_images = []
        if images is not None:
            images_path_full = os.path.join(folder_paths.get_input_directory(), images)
            offset = count_images_overlap
            images_from_file, fps, total_length = awesome_loader(images_path_full, start + offset, end)
            w = images_from_file.shape[2]
            h = images_from_file.shape[1]
            if images_overlap is not None:
                out_images.append(resize_image(images_overlap, w, h))
            out_images.append(images_from_file)

        # get masks chunk from input file and construct the output
        out_masks = []
        if masks is not None:
            masks_path_full = os.path.join(folder_paths.get_input_directory(), masks)
            offset = count_maskeditor + max(count_images_overlap, count_masks_overlap)
            imasks_from_file = awesome_loader(masks_path_full, start + offset, end)[0]
            masks_from_file = image_to_mask(imasks_from_file)
            if mask_maskeditor is not None:
                out_masks.append(resize_mask(mask_maskeditor, w, h))
            if masks_overlap is not None:
                out_masks.append(resize_mask(masks_overlap, w, h))
            else:
                black_panel = panelMask(w, h, 0)
                out_masks.extend([black_panel] * count_images_overlap)
            out_masks.append(resize_mask(masks_from_file, w, h))

        out_images_torch = None
        if len(out_images) > 0:
            out_images_torch = torch.cat(out_images)
            assert len(out_images_torch.shape) == 4, f"images are not rank 4 {out_images_torch.shape}"

        out_masks_torch = None
        if len(out_masks) > 0:
            out_masks_torch = torch.cat(out_masks)
            assert len(out_masks_torch.shape) == 3, f"masks are not rank 3 {out_masks_torch.shape}"

        this_chunk_length = max(
            len(out_images_torch) if out_images_torch is not None else 0,
            len(out_masks_torch) if out_masks_torch is not None else 0,
        )

        chunk_count = math.ceil((total_length - chunk_overlap) / (chunk_length - chunk_overlap))

        c = {
            "mode": mode,
            "chunk_length": chunk_length, # this_chunk_length?
            "chunk_overlap": chunk_overlap,
            "total_length": total_length,
            "chunk_count": chunk_count,
        }

        chunker_data = {
            "start_node_id": unique_id,
            "index": s["index"],
            "chunker_config": c,
            "fps": fps,
        }

        ui_values = {
            "output_label_values": {
                "images": len(out_images_torch) if out_images_torch is not None else 0,
                "masks": len(out_masks_torch) if out_masks_torch is not None else 0,
                "width": w,
                "height": h,
                "index": s["index"],
                "chunk_length": this_chunk_length,
                "chunk_overlap": chunk_overlap,
                "total_length": total_length,
                "chunk_count": chunk_count,
            },
        }

        log(f"Starting chunk {s["index"] + 1} of {c["chunk_count"]}...")
        log(f"with images {out_images_torch.shape}")
        log(f"with masks {out_masks_torch.shape}")

        return {
            "ui": {"values": [ui_values]},
            "result": (
                chunker_data,
                out_images_torch,
                out_masks_torch,
                w,
                h,
                s["index"],
                this_chunk_length,
                chunk_overlap,
                total_length,
                chunk_count,
            ),
        }

class ChunkerCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunker_data": ("CHUNKER_DATA", {"tooltip": "Connect chunker_data from Chunker node to here"}),
                "show_debug": ("BOOLEAN", {"default": True, "tooltip": "Show debug overlay in preview"}),
                "select_overlaps_from": (["this_chunk", "previous_chunk"], {"default": "this_chunk", "tooltip": "TODO"}),
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
        show_debug,
        select_overlaps_from,
        images=None,
        masks=None,
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
            "preview_chunks": [],
        }

        # figure out if we have completed all chunks
        is_done = d["index"] + 1 >= c["chunk_count"]
        start = 0 if select_overlaps_from == "this_chunk" else c["chunk_overlap"]
        end = None if select_overlaps_from == "previous_chunk" else -c["chunk_overlap"]

        # save new image chunk to file
        if images is not None:
            log("[debug] Combine -> saving images chunk...", end="")
            images_full_path = save_video(images, d["fps"], "video/chunker/tmp/chunk/image/chunk")[0]
            print("done")
            s["image_chunks"].append(images_full_path)

        # save new mask chunk to file
        if masks is not None:
            log("[debug] Combine -> saving masks chunk...", end="")
            masks_full_path = save_video(mask_to_image(masks), d["fps"], "video/chunker/tmp/chunk/masks/chunk")[0]
            print("done")
            s["mask_chunks"].append(masks_full_path)

        # create preview from inputs
        log("[debug] Combine -> creating preview chunk...", end="")
        preview = create_preview_video(images, masks, show_debug, d, c)
        print("done")

        # save new preview chunk to file
        log("[debug] Combine -> saving preview chunk...", end="")
        preview_full_path = save_video(preview, d["fps"], "video/chunker/tmp/chunk/preview/chunk")[0]
        print("done")
        s["preview_chunks"].append(preview_full_path)

        # combine all preview chunks, excluding the overlaps
        filename_prefix = "video/chunker/tmp/chunks/preview/chunks" if not is_done else "video/chunker/tmp/chunks/preview/complete"
        log("[debug] Combine -> load all previews...", end="")
        all_preview = tuple(map(lambda filename: awesome_loader(filename, start, end)[0], s["preview_chunks"]))
        all_preview_torch = torch.cat(all_preview)
        print("done")
        log("[debug] Combine -> save all previews together...", end="")
        all_preview_video_path = save_video(all_preview_torch, d["fps"], filename_prefix)[1]
        print("done")

        # if no more chunks needed, return early
        if is_done:
            # load all image chunks as tensor
            out_images_torch = None
            if len(s["image_chunks"]) > 0:
                log("[debug] Combine -> load all images...", end="")
                all_images = tuple(map(lambda filename: awesome_loader(filename, start, end)[0], s["image_chunks"]))
                out_images_torch = torch.cat(all_images)
                print("done")
                log("[debug] Combine -> save all images together...", end="")
                save_video(out_images_torch, d["fps"], "video/chunker/images")
                print("done")

            # load all mask chunks as tensor
            out_masks_torch = None
            if len(s["mask_chunks"]) > 0:
                log("[debug] Combine -> load all masks...", end="")
                all_masks = tuple(map(lambda filename: awesome_loader(filename, start, end)[0], s["masks_chunks"]))
                out_masks_torch = torch.cat(all_masks)
                print("done")
                log("[debug] Combine -> save all masks together...", end="")
                save_video(out_masks_torch, d["fps"], "video/chunker/images")
                print("done")

            ui_values = {
                "input_label_values": {
                    "images": len(images) if images is not None else 0,
                    "masks": len(masks) if masks is not None else 0,
                },
                "output_label_values": {
                    "images": len(out_images_torch) if out_images_torch is not None else 0,
                    "masks": len(out_masks_torch) if out_masks_torch is not None else 0,
                },
                "index": d["index"],
                "chunk_count": c["chunk_count"],
                "video_path": all_preview_video_path,
            }

            log(f"Finished all chunks {d["index"] + 1} of {c["chunk_count"]}!")

            return {
                "ui": {"values": [ui_values]},
                "result":(
                    out_images_torch,
                    image_to_mask(out_masks_torch),
                )
            }

        # clone all the nodes between Chunker and ChunkerCombine
        graph = GraphBuilder()
        comfyuiRepeatNodes(dynprompt, graph, unique_id, d["start_node_id"])

        # update the store in the new_chunker
        new_chunker = graph.lookup_node(d["start_node_id"])
        new_chunker.set_input("store", {
            "index": d["index"] + 1,
            "images_last_chunk_path": s["image_chunks"][-1] if len(s["image_chunks"]) > 0 else None, # filename of last image chunk saved
            "masks_last_chunk_path": s["mask_chunks"][-1] if len(s["mask_chunks"]) > 0 else None, # filename of last mask chunk saved
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
        new_combine.set_input("store", s)

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
            "video_path": all_preview_video_path,
        }

        log(f"Finished chunk {d["index"] + 1} of {c["chunk_count"]}")

        return {
            "ui": {"values": [ui_values]},
            "result": (
                new_combine.out(0),
                new_combine.out(1),
            ),
            "expand": graph.finalize(),
        }
