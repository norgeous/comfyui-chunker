import torch
import math
from datetime import datetime
#from tqdm import tqdm
from comfy_execution.graph_utils import GraphBuilder
from comfy_extras.nodes_video import CreateVideo, SaveVideo
from comfy.utils import ProgressBar
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
from .utils import log, panelImage, panelMask, slice, len2, resizeImage, resizeMask, overlay_debug
from .repeatNodes import comfyuiRepeatNodes, getNodeIdsByType

class Chunker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "tooltip": "Starting index. This should be hidden in the UI"}),
                "chunk_length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4, "tooltip": "Count of images in each chunk"}),
                "chunk_overlap": ("INT", {"default": 4, "min": 0, "max": 4096, "step": 1, "tooltip": "Count of images to overlap between chunks"}),
                "total_length": ("INT", {"default": 158, "min": 1, "max": 100000, "step": 1, "tooltip": "Minimum count of images in the final output"}),
                "aspect_ratio": (["keep_input", "stretch_to_new", "crop_to_new"], {"tooltip": "`keep_input` = use width and height as megapixel density and retain original aspect ratio\n`stretch_to_new` = stretch to exact size specified\n`crop_to_new` = scale and crop to exact specified size"}),
                "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the output control_video and control_masks"}),
                "height": ("INT", {"default": 464, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the output control_video and control_masks"}),
            },
            "optional": {
                "control_video": ("IMAGE", {"tooltip": "None, Single Image or Images to be chunked"}),
                "control_masks": ("MASK", {"tooltip": "None, Single Mask or Masks to be chunked"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("CHUNK_INFO", "IMAGE", "MASK", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("chunk_info", "control_video", "control_masks", "width", "height", "chunk_length", "index", "loop_count")
    OUTPUT_TOOLTIPS = (
        "Connect \"chunk_info\" to the \"ChunkerCombine\" node",
        "Chunk of control_video for WanVaceToVideo",
        "Chunk of control_masks for WanVaceToVideo",
        "Width of control_video and control_masks",
        "Height of control_video and control_masks",
        "The length of this chunk",
        "The current itteration index, ie; 0, 1, 2, ...",
        "Total count of chunks",
    )
    FUNCTION = "chunker_start"
    CATEGORY = "Chunker"
    DESCRIPTION = "Chunker!"

    def chunker_start(
        self,
        width,
        height,
        aspect_ratio,
        chunk_length,
        chunk_overlap,
        total_length,
        control_video=None,
        control_masks=None,
        index=0,
        unique_id=None,
    ):
        # calculate how many chunks we need to fill total_length
        loop_count = max(1, math.ceil((total_length - chunk_overlap) / (chunk_length - chunk_overlap)))

        log(f"Starting chunk {index + 1} of {loop_count}...")

        comfy_pbar = ProgressBar(loop_count)
        #comfy_pbar.update(1)
        comfy_pbar.update_absolute(index + 1)

        #tqdm_pbar = tqdm(loop_count, desc="CHUNKING?")
        #tqdm_pbar.update(1)

        # resize the control_video to input width and height using copy of Kijai's method
        control_video = resizeImage(control_video, width, height, aspect_ratio)

        # resize the control_masks to input width and height using copy of Kijai's method
        control_masks = resizeMask(control_masks, width, height, aspect_ratio)

        w = control_video.shape[2] if control_video is not None else width
        h = control_video.shape[1] if control_video is not None else height

        control_video_length = len(control_video) if control_video is not None else 0
        control_masks_length = len(control_masks) if control_masks is not None else 0

        # exclude overlap in first chunk
        adjusted_overlap = 0 if index == 0 else chunk_overlap
        adjusted_length = chunk_length - adjusted_overlap

        # determine where current chunk starts and ends
        overlap_start = index * adjusted_length
        chunk_start = overlap_start + adjusted_overlap
        after_start = chunk_start + adjusted_length

        # create control_video and create control_masks
        grey_panel  = panelImage(w, h, 127, 127, 127)
        black_panel = panelMask(w, h, 0)
        white_panel = panelMask(w, h, 255)
        control_video_chunk = []
        control_masks_chunk = []

        # copy cv overlap
        control_video_chunk.extend(slice(control_video, overlap_start, chunk_start))

        # fill black masks to match cv overlap
        control_masks_chunk.extend([black_panel] * len2(control_video_chunk))

        # copy cv chunk
        control_video_chunk.extend(slice(control_video, chunk_start, after_start))

        # if single image with no mask mode, add one black panel
        if control_video_length == 1 and control_masks_length == 0:
            control_masks_chunk.extend([black_panel])

        # copy control_masks
        control_masks_chunk.extend(slice(control_masks, chunk_start, after_start))

        # fill remaining cv length with grey panels
        control_video_chunk.extend([grey_panel] * (chunk_length - len2(control_video_chunk)))

        # fill remaining masks length with white panels
        control_masks_chunk.extend([white_panel] * (chunk_length - len2(control_masks_chunk)))

        # collect chunks to be sent to ChunkerCombine
        before = control_video[:overlap_start] if control_video is not None else None
        after = control_video[after_start:] if control_video is not None else None

        chunk_info = {
            "start_node_id": unique_id,
            "index": index,
            "loop_count": loop_count,
            "chunk_length": chunk_length,
            "chunk_overlap": chunk_overlap,
            "before": before,
            "after": after,
        }

        ui_values = {
            "output_label_values": {
                "width": w,
                "height": h,
                "chunk_length": chunk_length,
                "index": index,
                "loop_count": loop_count,
            },
        }

        return {
            "ui": {"values": [ui_values]},
            "result": (
                chunk_info,
                torch.cat(control_video_chunk), # just this chunk
                torch.cat(control_masks_chunk), # just this chunk
                w,
                h,
                chunk_length,
                index,
                loop_count,
            ),
        }


class ChunkerCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_info": ("CHUNK_INFO", {"tooltip": "Connect chunk_info from Chunker node to here"}),
                "images": ("IMAGE", {"tooltip": "Processed chunk of images"}),
                "preview_fps": ("FLOAT", {"default": 16, "min": 1, "max": 120, "step": 1, "tooltip": "The FPS of the preview video"}),
                "debug": ("BOOLEAN", {"default": True, "tooltip": "Show debug overlay in preview"}),
            },
            "optional": {
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = ("Combined images from all chunks",)
    FUNCTION = "chunker_end"
    CATEGORY = "Chunker"
    DESCRIPTION = "ChunkerCombine"
    OUTPUT_NODE = True

    def chunker_end(
        self,
        chunk_info,
        images,
        debug,
        preview_fps,
        dynprompt=None,
        unique_id=None,
    ):
        start_node_id = chunk_info["start_node_id"]
        index = chunk_info["index"]
        loop_count = chunk_info["loop_count"]
        chunk_length = chunk_info["chunk_length"]
        chunk_overlap = chunk_info["chunk_overlap"]
        before = chunk_info["before"]
        after = chunk_info["after"]

        log(f"Finished chunk {index + 1} of {loop_count}!")

        comfy_pbar = ProgressBar(loop_count)
        #comfy_pbar.update(1)
        comfy_pbar.update_absolute(index + 1)

        out_images = []
        if before is not None: out_images.extend([before])
        out_images.extend([images])
        image_count = len2(out_images)
        completed_images_torch = torch.cat(out_images, dim=0)

        # apply debug overlay
        preview_video_torch = overlay_debug(completed_images_torch, chunk_length, chunk_overlap) if debug else completed_images_torch

        # save preview video
        create_video_node = CreateVideo()
        video, = create_video_node.create_video(preview_video_torch, preview_fps)
        save_video_node = SaveVideo()
        save_to = "video/chunker/tmp/tmp" if index+1 != loop_count else 'video/chunker/tmp/complete'
        save_result = save_video_node.save_video(video, save_to, "auto", "auto")
        video_path = save_result["ui"]["images"][0]

        if index >= loop_count - 1:
            # We're done with the loop, return all completed chunks
            ui_values = {
                "output_label_values": {
                    "images": image_count,
                },
                "image_count": image_count,
                "index": index,
                "loop_count": loop_count,
                "video_path": video_path,
            }
            return {
                "ui": {"values": [ui_values]},
                "result":(completed_images_torch,)
            }

        # We want to continue looping

        # create a copy of the nodes between Chunker and ChunkerCombine
        graph = GraphBuilder()
        comfyuiRepeatNodes(dynprompt, graph, unique_id, start_node_id)

        # add the yet to be completed images back into the control_video that is sent back to the start of the loop
        if after is not None: out_images.extend([after])

        out_images_torch = torch.cat(out_images, dim=0)

        # set the updated inputs on the Chunker node
        new_chunker = graph.lookup_node(start_node_id)
        new_chunker.set_input("control_video", out_images_torch) # update the start node's control_video with copy which includes the new chunk

        # increment start node's index, so it knows which chunk is next
        new_chunker.set_input("index", index + 1)

        # increment seeds in cloned KSamplers, to prevent same motion in each chunk
        ids = getNodeIdsByType(graph.finalize(), "KSampler")
        for id in ids:
            real_id = id.replace(f"{unique_id}.0.0.", "")
            node = graph.lookup_node(real_id)
            seed = node.get_input("seed")
            node.set_input("seed", seed + index + 1)

        ui_values = {
            "output_label_values": {
                "images": None,
            },
            "image_count": image_count,
            "index": index,
            "loop_count": loop_count,
            "video_path": video_path,
        }

        my_clone = graph.lookup_node("Recurse")

        return {
            "ui": {"values": [ui_values]},
	    "result": (my_clone.out(0),),
            "expand": graph.finalize(),
        }
