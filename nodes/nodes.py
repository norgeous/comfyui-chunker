import os
import folder_paths
import torch
import math
from comfy_execution.graph_utils import GraphBuilder
from .utils import count, log, panel_image, panel_mask, mask_to_image, image_to_mask, resize_image, resize_mask, get_input_filenames, force_wan_length, fix_total_length, get_this_chunk_length
from .debug_overlay import create_preview_video
from .repeatNodes import comfyuiRepeatNodes, getNodeIdsByType
from .loader import awesome_loader, quick_combine, save_video
from .loadAudio import load_audio

class Chunker:
    @classmethod
    def INPUT_TYPES(cls):
        files = ["None", *sorted(get_input_filenames())]
        return {
            "required": {
                "mode": (["None", "Wan"], {"tooltip": "TODO"}),
                "chunk_length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 1, "tooltip": "Count of images in each chunk"}),
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
    RETURN_NAMES = ("chunker_data", "images", "masks", "width", "height", "chunk_length", "chunk_overlap", "total_length", "chunk_count", "index")
    OUTPUT_TOOLTIPS = (
        "Connect \"chunker_data\" to the \"ChunkerCombine\" node",
        "Chunk of images",
        "Chunk of masks",
        "Width of images",
        "Height of images",
        "Count of images in each chunk",
        "Count of images to overlap between each chunk",
        "Total length of output images",
        "Count of chunks",
        "The current itteration index, ie; 0, 1, 2, ...",
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

        w = 512
        h = 512
        fps = 30

        if mode == "Wan":
            chunk_length = force_wan_length(chunk_length)
            total_length = fix_total_length(total_length, chunk_length, chunk_overlap)
            fps = 16

        #this_chunk_length = total_length - s["index"] * (chunk_length - chunk_overlap)
        this_chunk_length = get_this_chunk_length(s["index"], chunk_length, chunk_overlap, total_length)
        log("this_chunk_length", this_chunk_length)

        start = s["index"] * (chunk_length - chunk_overlap)
        end = start + chunk_length
        chunk_count = math.ceil((total_length - chunk_overlap) / (chunk_length - chunk_overlap))

        if images == "None": images = None
        if masks == "None": masks = None
        if image == "None": image = None
        if image_paint == "None": image_paint = None

        black_panel = panel_mask(w, h, 0)

        out_images = []
        out_masks = []

        # get the images overlap from store file
        if s["images_last_chunk_path"] is not None:
            images_overlap = awesome_loader(s["images_last_chunk_path"], start=-chunk_overlap)[0]
            w = images_overlap.shape[2]
            h = images_overlap.shape[1]
            out_images.append(images_overlap)
            if mode == "Wan":
                out_masks.append(torch.cat([black_panel] * len(images_overlap))) # add same amount of black masks to masks

        # get images chunk from input file
        if images is not None:
            images_path_full = os.path.join(folder_paths.get_input_directory(), images)
            images_chunk, images_fps, images_total_length = awesome_loader(images_path_full, start + count(out_images), end)
            if images_fps is not None: fps = images_fps
            if total_length == 0: total_length = images_total_length
            w = images_chunk.shape[2]
            h = images_chunk.shape[1]
            if images_total_length > 1 or images_total_length == 1 and s["index"]==0: out_images.append(images_chunk)
            if mode == "Wan" and images_total_length == 1 and s["index"]==0:
                out_masks.append(black_panel) # add 1 black mask to masks (for i2v)

        # get the mask from the mask editor for first chunk only
        if image is not None and s["index"] == 0:
            #perhaps use folder_paths.get_annotated_filepath(image)
            if " [input]" in image:
                mask_editor_filename = image.replace("clipspace/", "").replace(" [input]", "")
                path_full = os.path.join(folder_paths.get_input_directory(), 'clipspace', mask_editor_filename)
            if " [temp]" in image:
                mask_editor_filename = image.replace(" [temp]", "")
                path_full = os.path.join(folder_paths.get_temp_directory(), mask_editor_filename)
            mask_maskeditor = awesome_loader(path_full)[0]
            out_masks.append(mask_maskeditor)

        # get the masks overlap from store file
        if s["masks_last_chunk_path"] is not None:
            imasks_overlap = awesome_loader(s["masks_last_chunk_path"], start=-chunk_overlap)[0]
            masks_overlap = image_to_mask(imasks_overlap)
            out_masks.append(masks_overlap)

        # get masks chunk from input file
        if masks is not None:
            masks_path_full = os.path.join(folder_paths.get_input_directory(), masks)
            imasks_chunk = awesome_loader(masks_path_full, start + count(out_masks), end)[0]
            masks_chunk = image_to_mask(imasks_chunk)
            out_masks.append(masks_chunk)

        # do some stuff for Wan
        if mode == "Wan":
             grey_panel = panel_image(w, h, 128, 128, 128)
             white_panel = panel_mask(w, h, 255)

             # if no images invent some blank (grey) ones (for t2v)
             if count(out_images) < this_chunk_length: out_images.append(torch.cat([grey_panel] * (this_chunk_length - count(out_images))))

             # if no masks invent some blank (white) ones (for t2v)
             if count(out_masks) < this_chunk_length: out_masks.append(torch.cat([white_panel] * (this_chunk_length - count(out_masks))))

        out_images_torch = None
        if len(out_images) > 0:
            out_images_resized = list(map(lambda tensor: resize_image(tensor, w, h), out_images))
            out_images_torch = torch.cat(out_images_resized)
            assert len(out_images_torch.shape) == 4, f"images are not rank 4 {out_images_torch.shape}"

        out_masks_torch = None
        if len(out_masks) > 0:
            out_masks_resized = list(map(lambda tensor: resize_mask(tensor, w, h), out_masks))
            out_masks_torch = torch.cat(out_masks_resized)
            assert len(out_masks_torch.shape) == 3, f"masks are not rank 3 {out_masks_torch.shape}"

        c = {
            "mode": mode,
            "chunk_length": chunk_length,
            "chunk_overlap": chunk_overlap,
            "total_length": total_length,
            "chunk_count": chunk_count,
        }

        chunker_data = {
            "start_node_id": unique_id,
            "index": s["index"],
            "chunker_config": c,
            "audio": images if images is not None and images.endswith(".mp4") else None,
            "fps": fps,
        }

        ui_values = {
            "output_label_values": {
                "images": count(out_images),
                "masks": count(out_masks),
                "width": w,
                "height": h,
                "chunk_length": max(count(out_images), count(out_masks)),
                "chunk_overlap": chunk_overlap,
                "total_length": total_length,
                "chunk_count": chunk_count,
                "index": s["index"],
            },
        }

        log(f"Starting chunk {s["index"] + 1} of {c["chunk_count"]}...")

        return {
            "ui": {"values": [ui_values]},
            "result": (
                chunker_data,
                out_images_torch,
                out_masks_torch,
                w,
                h,
                max(count(out_images), count(out_masks)),
                chunk_overlap,
                total_length,
                chunk_count,
                s["index"],
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

        # save new image chunk to a new file
        if images is not None:
            #log("[debug] Combine -> saving images chunk...", end="")
            images_full_path = save_video(images, d["fps"], "video/chunker/tmp/chunk/image/chunk")[0]
            s["image_chunks"].append(images_full_path)
            #print("done")

        # save new mask chunk to a new file
        if masks is not None:
            #log("[debug] Combine -> saving masks chunk...", end="")
            masks_full_path = save_video(mask_to_image(masks), d["fps"], "video/chunker/tmp/chunk/masks/chunk")[0]
            s["mask_chunks"].append(masks_full_path)
            #print("done")

        # create preview from inputs
        #log("[debug] Combine -> creating preview chunk...", end="")
        preview = create_preview_video(images, masks, show_debug, d, c)
        #print("done")

        # save new preview chunk to a new file
        #log("[debug] Combine -> saving preview chunk...", end="")
        preview_full_path = save_video(preview, d["fps"], "video/chunker/tmp/chunk/preview/chunk")[0]
        s["preview_chunks"].append(preview_full_path)
        #print("done")

        # combine all preview chunks to a new file, excluding the overlaps
        #log("[debug] Combine -> combine all previews...", end="")
        filename_prefix = "video/chunker/tmp/chunks/preview/chunks" if not is_done else "video/chunker/tmp/chunks/preview/complete"
        all_preview_frontend_data = quick_combine(s["preview_chunks"], c["chunk_overlap"], select_overlaps_from, filename_prefix)[1]
        #print("done")

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

            ui_values = {
                "input_label_values": {
                    "images": len(images) if images is not None else 0,
                    "masks": len(masks) if masks is not None else 0,
                },
                "output_label_values": {
                    "images": len(out_images_torch) if out_images_torch is not None else 0,
                    "masks": len(out_masks_torch) if out_masks_torch is not None else 0,
                    "fps": f"{d["fps"]:.2f}",
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
                    load_audio(d["audio"]) if d["audio"] is not None else None,
                    d["fps"],
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
