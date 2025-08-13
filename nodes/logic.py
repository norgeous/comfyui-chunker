from PIL import Image
import torch
import numpy as np
from comfy_execution.graph_utils import GraphBuilder, is_link
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS

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

class Chunker:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4, "tooltip": "Count of frames in each chunk"}),
                "overlap": ("INT", {"default": 2, "min": 0, "max": 4096, "step": 1, "tooltip": "Count of frames to overlap between chunks"}),
                "loop_count": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1, "tooltip": "Count of itterations"}),
            },
            "optional": {
                "control_video": ("IMAGE", {"tooltip": "None, Single Image or Images"}),
                "control_masks": ("MASK", {"tooltip": "None, Single Mask or Masks"}),
                "width": ("INT", {"defaultInput": True, "tooltip": "Width fallback, used if no image provided. (default: 512)"}),
                "height": ("INT", {"defaultInput": True, "tooltip": "Height fallback, used if no image provided. (default: 512)"}),
                "index": ("INT", {"tooltip": "Starting index. Leave this as 0"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("CHUNKER_FLOW_CONTROL", "IMAGE", "CHUNK_INFO", "IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("chunker_flow", "previous_chunks", "chunk_info", "control_video", "control_masks", "width", "height", "length", "index")
    OUTPUT_TOOLTIPS = (
        "Connect chunker_flow to the ChunkerCombine node",
        "All images from previous chunks or None",
        "Connect **chunk_info** to the **ChunkerCombine** node",
        "Chunk of control_video for WanVaceToVideo",
        "Chunk of control_masks for WanVaceToVideo",
        "Width of control_video and control_masks",
        "Height of control_video and control_masks",
        "The length of this chunk",
        "The current itteration index, ie; 0, 1, 2, ...",
    )
    FUNCTION = "chunker_start"
    CATEGORY = "Chunker"

    def chunker_start(
        self,
        length,
        overlap,
        loop_count,
        control_video=None,
        control_masks=None,
        width=512,
        height=512,
        index=0,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
        **kwargs
    ):
        images_before = None
        images_overlap = None
        images_after = None
        control_video_chunk = None

        adjusted_overlap = 0 if index == 0 else overlap # exclude overlap in first chunk
        adjusted_length = length - adjusted_overlap

        w = control_video.shape[2] if control_video is not None else width
        h = control_video.shape[1] if control_video is not None else height

        overlap_start = index * adjusted_length
        chunk_start = overlap_start + adjusted_overlap
        after_start = chunk_start + adjusted_length

        print("overlap_start", overlap_start)
        print("chunk_start", chunk_start)
        print("after_start", after_start)

        previous_chunks = control_video[:overlap_start] if control_video is not None else None

        # create control_video and create control_masks
        grey_panel  = pil2tensor(Image.new('RGB', (w, h), (128, 128, 128)))
        black_panel = pil2tensor(Image.new('RGB', (w, h), (0,   0,   0  )).convert('L'))
        white_panel = pil2tensor(Image.new('RGB', (w, h), (255, 255, 255)).convert('L'))

        control_video_chunk = []
        control_masks_chunk = []

        # copy cv chunk
        control_video_chunk.extend(slice(control_video, overlap_start, chunk_start))
        control_video_chunk.extend(slice(control_video, chunk_start, after_start))

	# copy masks chunk
        control_masks_chunk.extend(slice(control_masks, overlap_start, chunk_start))
        control_masks_chunk.extend(slice(control_masks, chunk_start, after_start))

        # fill masks to match cv
        control_masks_chunk.extend([black_panel] * (len2(control_video_chunk) - len2(control_masks_chunk)))

        # fill remaining length with grey panels
        control_video_chunk.extend([grey_panel] * (length - len2(control_video_chunk)))

        # fill remaining length with white panels
        control_masks_chunk.extend([white_panel] * (length - len2(control_masks_chunk)))

        print("video len2:", len2(control_video_chunk))
        print("masks len2:", len2(control_masks_chunk))

        chunk_info = {
            "start_node_id": unique_id,
            "original_control_video": control_video,
            "previous_chunks": previous_chunks,
        }

        return (
            "chunker_flow_stub",
            previous_chunks,
            chunk_info,
            torch.cat(control_video_chunk),
            torch.cat(control_masks_chunk),
            w,
            h,
            length,
            index,
        )


class ChunkerCombine:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunker_flow": ("CHUNKER_FLOW_CONTROL", {"rawLink": True, "tooltip": "Connect chunker_flow from Chunker node to here"}),
                "previous_chunks": ("IMAGE", {"tooltip": "Connect previous_chunks from Chunker node to here"}),
                "chunk_info": ("CHUNK_INFO", {"tooltip": "Connect **chunk_info** from **Chunker** node to here"}),
                "images": ("IMAGE", {"tooltip": "Processed chunk of images"}),
            },
            "optional": {
            },
            "hidden": {
                "prompt": "PROMPT",
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = ("Combined images from all chunks",)
    FUNCTION = "chunker_end"
    CATEGORY = "Chunker"

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

    def chunker_end(self, chunker_flow, previous_chunks, chunk_info, images, prompt=None, dynprompt=None, unique_id=None, extra_pnginfo=None):
        #print("got chunk_info", chunk_info)
        start_node_id = chunk_info["start_node_id"]
        original_control_video = chunk_info["original_control_video"]
        previous_chunks = chunk_info["previous_chunks"]

        forstart_node = dynprompt.get_node(start_node_id)
        loop_count = forstart_node["inputs"]["loop_count"]
        index = forstart_node["inputs"]["index"]

        new_images = []
        if previous_chunks is not None: new_images.extend([previous_chunks])
        new_images.extend([images])
        if original_control_video is not None: new_images.extend(slice(original_control_video, len2(new_images), None))

        #print("len2 new_images", len2(new_images))
        #print("len2 slice of cv", len2(slice(original_control_video, len2(new_images), None)))
        #print("original_control_video len:", len(original_control_video))

        new_images_torch = torch.cat(new_images, dim=0)

        if index >= loop_count - 1:
            # We're done with the loop
            return (new_images_torch,)

        print("starting repetition ", index + 1, " of ", loop_count - 1, "...")

        # We want to loop
        this_node = dynprompt.get_node(unique_id)
        upstream = {}

        # Get the list of all nodes between the open and close nodes
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
        open_node = start_node_id # chunker_flow[0]
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
