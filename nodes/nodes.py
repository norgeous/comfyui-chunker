import torch
import math
from comfy_execution.graph_utils import GraphBuilder, is_link
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
from .utils import panelImage, panelMask, slice, len2, resizeImage, resizeMask
from .repeatnodes import comfyuiRepeatNodes

class Chunker:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "tooltip": "Starting index. This should be hidden in the UI"}),
                "width": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the output control_video and control_masks"}),
                "height": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the output control_video and control_masks"}),
                "aspect_ratio": (["keep_input", "stretch_to_new", "crop_to_new"], {"tooltip": "`keep_input` = use width and height as megapixel density and retain original aspect ratio\n`stretch_to_new` = stretch to exact size specified\n`crop_to_new` = scale and crop to exact specified size"}),
                "chunk_length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4, "tooltip": "Count of images in each chunk"}),
                "chunk_overlap": ("INT", {"default": 4, "min": 0, "max": 4096, "step": 1, "tooltip": "Count of images to overlap between chunks"}),
                "total_length": ("INT", {"default": 158, "min": 1, "max": 100000, "step": 1, "tooltip": "Count of images in the final output"}),
            },
            "optional": {
                "control_video": ("IMAGE", {"tooltip": "None, Single Image or Images to be chunked"}),
                "control_masks": ("MASK", {"tooltip": "None, Single Mask or Masks to be chunked"}),
            },
            "hidden": {
                # "prompt": "PROMPT",
                # "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
                # "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("CHUNK_INFO", "IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("chunk_info", "control_video", "control_masks", "width", "height", "chunk_length", "index")
    OUTPUT_TOOLTIPS = (
        "Connect \"chunk_info\" to the \"ChunkerCombine\" node",
        "Chunk of control_video for WanVaceToVideo",
        "Chunk of control_masks for WanVaceToVideo",
        "Width of control_video and control_masks",
        "Height of control_video and control_masks",
        "The length of this chunk",
        "The current itteration index, ie; 0, 1, 2, ...",
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
        # prompt=None,
        # dynprompt=None,
        unique_id=None,
        # extra_pnginfo=None,
    ):
        # calculate how many chunks we need to fill total_length
        loop_count = math.ceil((total_length - chunk_overlap) / (chunk_length - chunk_overlap))

        print(f"\U0001F36B  CHUNKER: Starting chunk {index + 1} of {loop_count}...")

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

        # collect previous chunks to be sent to ChunkerCombine
        previous_chunks = control_video[:overlap_start] if control_video is not None else None

        chunk_info = {
            "start_node_id": unique_id,
            "index": index,
            "loop_count": loop_count,
            "original_control_video": control_video,
            "previous_chunks": previous_chunks,
        }

        ui_values = {
            "width": w,
            "height": h,
            "chunk_length": chunk_length,
            "index": index,
            "loop_count": loop_count,
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
            ),
        }


class ChunkerCombine:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_info": ("CHUNK_INFO", {"tooltip": "Connect chunk_info from Chunker node to here"}),
                "images": ("IMAGE", {"tooltip": "Processed chunk of images"}),
            },
            "optional": {
            },
            "hidden": {
                # "prompt": "PROMPT",
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
                # "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = ("Combined images from all chunks",)
    FUNCTION = "chunker_end"
    CATEGORY = "Chunker"
    DESCRIPTION = "ChunkerCombine"

    def chunker_end(
        self,
        chunk_info,
        images,
        # prompt=None,
        dynprompt=None,
        unique_id=None,
        # extra_pnginfo=None
    ):
        start_node_id = chunk_info["start_node_id"]
        loop_count = chunk_info["loop_count"]
        original_control_video = chunk_info["original_control_video"]
        previous_chunks = chunk_info["previous_chunks"]

        forstart_node = dynprompt.get_node(start_node_id)
        index = forstart_node["inputs"]["index"]

        new_images = []
        if previous_chunks is not None: new_images.extend([previous_chunks])
        new_images.extend([images])
        completed_images_torch = torch.cat(new_images, dim=0)

        print(f"\U0001F36B  CHUNKER: Finished chunk {index + 1} of {loop_count}!")

        if index >= loop_count - 1:
            # We're done with the loop, return all completed chunks so far
            return (completed_images_torch,)

        # We want to continue looping

        # add the yet to be completed images back into the control_video that is sent back to the start of the loop
        if original_control_video is not None: new_images.extend(slice(original_control_video, len2(new_images), None))
        new_images_torch = torch.cat(new_images, dim=0)

        # create a copy of the nodes between Chunker and ChunkerCombine
        graph = GraphBuilder()
        comfyuiRepeatNodes(dynprompt, graph, unique_id, start_node_id)

        # set the updated inputs on the Chunker node
        new_open = graph.lookup_node(start_node_id)
        new_open.set_input("control_video", new_images_torch) # update the start node's control_video with copy which includes the new chunk
        new_open.set_input("index", index + 1) # increment start node's index, so it knows which chunk is next
        my_clone = graph.lookup_node("Recurse")

        return {
            "result": (my_clone.out(0),),
            "expand": graph.finalize(),
        }
