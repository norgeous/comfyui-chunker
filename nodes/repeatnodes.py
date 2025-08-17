from comfy_execution.graph_utils import GraphBuilder, is_link
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS

def explore_dependencies(node_id, dynprompt, upstream, parent_ids):
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
                explore_dependencies(parent_id, dynprompt, upstream, parent_ids)

            upstream[parent_id].append(node_id)

def explore_output_nodes(dynprompt, upstream, output_nodes, parent_ids):
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

def collect_contained(node_id, upstream, contained):
    if node_id not in upstream:
        return
    for child_id in upstream[node_id]:
        if child_id not in contained:
            contained[child_id] = True
            collect_contained(child_id, upstream, contained)


def comfyuiRepeatNodes(dynprompt, graph, unique_id, start_node_id):
    # Get the list of all nodes between the open and close nodes
    upstream = {}
    parent_ids = []
    explore_dependencies(unique_id, dynprompt, upstream, parent_ids)
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

    explore_output_nodes(dynprompt, upstream, output_nodes, parent_ids)
    contained = {}
    collect_contained(start_node_id, upstream, contained)
    contained[unique_id] = True
    contained[start_node_id] = True

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
