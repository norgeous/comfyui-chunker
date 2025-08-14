from PIL import Image
import torch
import numpy as np
from comfy_execution.graph_utils import GraphBuilder, is_link
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
from comfy.utils import common_upscale
import math

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def slice(thing, start=None, end=None):
    if thing is None: return []
    sliced = thing[start:end]
    if len(sliced) == 0: return []
    return [sliced]

def len2(thing):
    count = 0
    for item in thing:
        count += len(item)
    return count

def kijaiWanResize(image, generation_width, generation_height, aspect_ratio_preservation):
    VAE_STRIDE = (4, 8, 8)
    PATCH_SIZE = (1, 2, 2)
    H, W = image.shape[1], image.shape[2]
    max_area = generation_width * generation_height
    crop = "disabled"
    if aspect_ratio_preservation == "keep_input":
        aspect_ratio = H / W
    elif aspect_ratio_preservation == "stretch_to_new" or aspect_ratio_preservation == "crop_to_new":
        aspect_ratio = generation_height / generation_width
        if aspect_ratio_preservation == "crop_to_new":
            crop = "center"
    lat_h = round(
    np.sqrt(max_area * aspect_ratio) // VAE_STRIDE[1] //
    PATCH_SIZE[1] * PATCH_SIZE[1])
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // VAE_STRIDE[2] //
        PATCH_SIZE[2] * PATCH_SIZE[2])
    h = lat_h * VAE_STRIDE[1]
    w = lat_w * VAE_STRIDE[2]
    resized_image = common_upscale(image.movedim(-1, 1), w, h, "lanczos", crop).movedim(1, -1)
    return resized_image

class Chunker:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the output control_video and control_masks"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the output control_video and control_masks"}),
                "aspect_ratio_preservation": (["keep_input", "stretch_to_new", "crop_to_new"],),
                "chunk_length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4, "tooltip": "Count of images in each chunk"}),
                "chunk_overlap": ("INT", {"default": 4, "min": 0, "max": 4096, "step": 1, "tooltip": "Count of images to overlap between chunks"}),
                "total_length": ("INT", {"default": 158, "min": 1, "max": 100000, "step": 1, "tooltip": "Count of images in the final output"}),
                "index": ("INT", {"tooltip": "Starting index. Leave this as 0"}),
            },
            "optional": {
                "control_video": ("IMAGE", {"tooltip": "None, Single Image or Images"}),
                "control_masks": ("MASK", {"tooltip": "None, Single Mask or Masks"}),
            },
            "hidden": {
                # "prompt": "PROMPT",
                # "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
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
        aspect_ratio_preservation,
        chunk_length,
        chunk_overlap,
        total_length,
        index,
        control_video=None,
        control_masks=None,
        # prompt=None,
        # extra_pnginfo=None,
        unique_id=None,
    ):
        # calculate how many chunks we need to fill total_length
        loop_count = math.ceil((total_length - chunk_overlap) / (chunk_length - chunk_overlap))

        print(f"🍫  CHUNKER: Starting chunk {index + 1} of {loop_count}...")

        # resize the input video to input width and height using copy of Kijai's method
        control_video = kijaiWanResize(control_video, width, height, aspect_ratio_preservation) if control_video is not None else None

        control_video_length = len(control_video) if control_video is not None else 0
        control_masks_length = len(control_masks) if control_masks is not None else 0

        adjusted_overlap = 0 if index == 0 else chunk_overlap # exclude overlap in first chunk
        adjusted_length = chunk_length - adjusted_overlap

        w = control_video.shape[2] if control_video is not None else width
        h = control_video.shape[1] if control_video is not None else height

        overlap_start = index * adjusted_length
        chunk_start = overlap_start + adjusted_overlap
        after_start = chunk_start + adjusted_length

        #print("overlap_start", overlap_start)
        #print("chunk_start", chunk_start)
        #print("after_start", after_start)

        previous_chunks = control_video[:overlap_start] if control_video is not None else None

        # create control_video and create control_masks
        grey_panel  = pil2tensor(Image.new('RGB', (w, h), (128, 128, 128)))
        black_panel = pil2tensor(Image.new('RGB', (w, h), (0,   0,   0  )).convert('L'))
        white_panel = pil2tensor(Image.new('RGB', (w, h), (255, 255, 255)).convert('L'))

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

        chunk_info = {
            "start_node_id": unique_id,
            "index": index,
            "loop_count": loop_count,
            "original_control_video": control_video,
            "previous_chunks": previous_chunks,
        }

        return (
            chunk_info,
            torch.cat(control_video_chunk),
            torch.cat(control_masks_chunk),
            w,
            h,
            chunk_length,
            index,
        )


class ChunkerCombine:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_info": ("CHUNK_INFO", {"tooltip": "Connect **chunk_info** from **Chunker** node to here"}),
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

    def explore_dependencies(self, node_id, dynprompt, upstream, parent_ids):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return

        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                display_id = dynprompt.get_display_node_id(parent_id)
                display_node = dynprompt.get_node(display_id)
                class_type = display_node["class_type"]
                if class_type not in ['ChunkerCombine', 'easy forLoopEnd', 'easy whileLoopEnd']:
                    parent_ids.append(display_id)
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(parent_id, dynprompt, upstream, parent_ids)

                upstream[parent_id].append(node_id)

    def explore_output_nodes(self, dynprompt, upstream, output_nodes, parent_ids):
        for parent_id in upstream:
            display_id = dynprompt.get_display_node_id(parent_id)
            for output_id in output_nodes:
                id = output_nodes[output_id][0]
                if id in parent_ids and display_id == id and output_id not in upstream[parent_id]:
                    if '.' in parent_id:
                        arr = parent_id.split('.')
                        arr[len(arr)-1] = output_id
                        upstream[parent_id].append('.'.join(arr))
                    else:
                        upstream[parent_id].append(output_id)

    def collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)

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

        print(f"🍫  CHUNKER: Finished chunk {index + 1} of {loop_count}!")

        if index >= loop_count - 1:
            # We're done with the loop
            return (completed_images_torch,)

        # We want to loop

        # add the yet to be completed images back into the control_video that is sent back to the start of the loop
        if original_control_video is not None: new_images.extend(slice(original_control_video, len2(new_images), None))
        new_images_torch = torch.cat(new_images, dim=0)

        # Get the list of all nodes between the open and close nodes
        upstream = {}
        parent_ids = []
        self.explore_dependencies(unique_id, dynprompt, upstream, parent_ids)
        parent_ids = list(set(parent_ids))

        # Get the list of all output nodes between the open and close nodes
        prompts = dynprompt.get_original_prompt()
        output_nodes = {}
        for id in prompts:
            node = prompts[id]
            if "inputs" not in node:
                continue
            class_type = node["class_type"]
            class_def = ALL_NODE_CLASS_MAPPINGS[class_type]
            if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
                for k, v in node['inputs'].items():
                    if is_link(v):
                        output_nodes[id] = v

        self.explore_output_nodes(dynprompt, upstream, output_nodes, parent_ids)
        contained = {}
        open_node = start_node_id
        self.collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        graph = GraphBuilder()

        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)

        new_open = graph.lookup_node(open_node)
        new_open.set_input("control_video", new_images_torch)
        new_open.set_input("index", index + 1)
        my_clone = graph.lookup_node("Recurse")

        return {
            "result": (my_clone.out(0),),
            "expand": graph.finalize(),
        }
