from .utils import AlwaysEqualProxy, ByPassTypeTuple, compare_revision
from comfy_execution.graph_utils import GraphBuilder, is_link
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS

DEFAULT_FLOW_NUM = 2
MAX_FLOW_NUM = 2
lazy_options = {"lazy": True} if compare_revision(2543) else {}

any_type = AlwaysEqualProxy("*")

class forLoopStart:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4, "tooltip": "Count of frames in each chunk"}),
                "overlap": ("INT", {"default": 2, "min": 0, "max": 4096, "step": 1, "tooltip": "Count of frames to overlap between chunks"}),
                "loop_count": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
            },
            "optional": {
                "control_video": ("IMAGE", {"tooltip": "None, Single Image or Images"}),
                "control_masks": ("MASK", {"tooltip": "None, Single Mask or Masks"}),
                "width": ("INT", {"defaultInput": True, "tooltip": "Width fallback, used if no image provided. (default: 500)"}),
                "height": ("INT", {"defaultInput": True, "tooltip": "Height fallback, used if no image provided. (default: 500)"}),
                "index": ("INT", {"tooltip": ""}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("CHUNKER_FLOW_CONTROL", "IMAGE", "MASK", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("chunker_flow", "control_video", "control_masks", "width", "height", "length", "index",)
    FUNCTION = "for_loop_start"
    CATEGORY = "Chunker"

    def for_loop_start(
        self,
        length,
        overlap,
        loop_count,
        control_video=None,
        control_masks=None,
        width=500,
        height=500,
        index=0,
        prompt=None,
        extra_pnginfo=None,
        unique_id=None,
        **kwargs
    ):
        control_video_output = control_video
        control_masks_output = control_masks

        return (
            "chunker_flow_stub",
            control_video_output,
            control_masks_output,
            width,
            height,
            length,
            index,
        )

        return {
            "result": ("chunker_flow_stub", control_video_output, control_masks_output, width, height, length, i),
            "expand": graph.finalize(),
        }







class forLoopEnd:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunker_flow": ("CHUNKER_FLOW_CONTROL", {"rawLink": True}),
                "images": ("IMAGE", {"rawLink": True}),
            },
            "optional": {
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "debug")
    FUNCTION = "for_loop_end"
    CATEGORY = "Chunker"

    def for_loop_end(self, chunker_flow, images, dynprompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        print("chunker_flow", chunker_flow)
        print("kwargs", kwargs)

        graph = GraphBuilder()
        forstart_node_id = chunker_flow[0]
        total = None

        # Using dynprompt to get the original node
        forstart_node = dynprompt.get_node(forstart_node_id)

        #print(forstart_node)

        if forstart_node['class_type'] == 'ChunkerForLoopStart':
            inputs = forstart_node['inputs']
            total = inputs['loop_count']

        print("total", total)

        #sub = graph.node("easy mathInt", operation="add", a=[forstart_node_id, 1], b=1)
        #cond = graph.node("easy compare", a=sub.out(0), b=total, comparison='a < b')
        #input_values = {("initial_value%d" % i): kwargs.get("initial_value%d" % i, None) for i in range(1, MAX_FLOW_NUM)}
        #while_close = graph.node("easy whileLoopEnd", flow=chunker_flow, condition=cond.out(0), initial_value0=sub.out(0), **input_values)

        #print("sub.out0", sub.out(0))
        #print("cond.out0", cond.out(0))
        #print("input_values", input_values)
        #print("while_close.out0", while_close.out(0))
        #print("while_close.out1", while_close.out(1))

        #index = graph.node("PrimitiveInt", value=0, control_after_generate="fixed")
        index = graph.node("PrimitiveString", value="1")
        cond = graph.node("StringCompare", string_a=index.out(0), string_b=str(total), mode="Equal", case_sensitive=False)
        print("index out", index.out(0))
        print("cond out", cond.out(0))

        print("nodes:", graph.nodes)

        #result = tuple([while_close.out(i) for i in range(1, MAX_FLOW_NUM)])
        images_out = images

        print("finalized:", graph.finalize())


        return {
            "result": (images_out, cond.out(0)),
            "expand": graph.finalize(),
        }






class ChunkerCombine2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunker_flow": ("CHUNKER_FLOW_CONTROL", {"rawLink": True}),
                #"images": ("IMAGE", {"rawLink": True}),
                "images": ("IMAGE",),
            },
            "optional": {
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("IMAGE", "*")
    RETURN_NAMES = ("images", "debug")
    FUNCTION = "while_loop_close"
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
                if class_type not in ['easy forLoopEnd', 'easy whileLoopEnd']:
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

    def while_loop_close(self, chunker_flow, images, dynprompt=None, unique_id=None, extra_pnginfo=None):
        forstart_node = dynprompt.get_node(chunker_flow[0])
        loop_count = forstart_node["inputs"]["loop_count"]
        index = forstart_node["inputs"]["index"]

        #images2 = dynprompt.get_node(images[0])

        if index >= loop_count - 1:
            #print("DONE!!!!!!!!!!!!!!!!!!!!", images)
            # We're done with the loop
            return (images,)

        print("starting repetition ", index + 1, " of ", loop_count - 1, "...")

        # We want to loop
        # this_node = dynprompt.get_node(unique_id)
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
        open_node = chunker_flow[0]
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
        new_open.set_input("control_video", images)
        new_open.set_input("index", index + 1)
        my_clone = graph.lookup_node("Recurse")

        return {
            "result": (my_clone.out(0),'hello'),
            "expand": graph.finalize(),
        }
