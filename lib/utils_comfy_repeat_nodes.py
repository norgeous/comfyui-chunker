from comfy_execution.graph_utils import GraphBuilder, is_link
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS


def get_parent_ids(dynprompt, node_id):
    node_info = dynprompt.get_node(node_id)
    if "inputs" not in node_info:
        return []
    parent_ids = []
    for _k, v in node_info["inputs"].items():
        if is_link(v):
            parent_id = v[0]
            grandparent_ids = get_parent_ids(dynprompt, parent_id)
            parent_ids.append([parent_id, *grandparent_ids])
    return parent_ids


def extract_chains(node, chain=None):
    if chain is None:
        chain = []
    if isinstance(node, str):
        return [chain + [node]]
    node_val = node[0]
    new_chain = chain + [node_val]
    children = node[1:]
    if not children:
        return [new_chain]
    paths = []
    for child in children:
        paths.extend(extract_chains(child, new_chain))
    return paths


def get_parent_id_chains(dynprompt, node_id):
    return extract_chains([node_id, *get_parent_ids(dynprompt, node_id)])


def get_all_output_node_ids(dynprompt):
    prompt = dynprompt.ephemeral_prompt if len(dynprompt.ephemeral_prompt) > 0 else dynprompt.get_original_prompt()
    return [
        id for id,
        info in prompt.items() if getattr(ALL_NODE_CLASS_MAPPINGS.get(info.get("class_type")), "OUTPUT_NODE", False) is True
    ]


def get_ids_by_partial_names(dynprompt, partial_names):
    prompt = dynprompt.ephemeral_prompt if len(dynprompt.ephemeral_prompt) > 0 else dynprompt.get_original_prompt()
    return [id for partial in partial_names for id, info in prompt.items() if partial in info["class_type"]]


def get_ids_by_partial_names_in_graph(graph, partial_names):
    return [id for id, node in graph.finalize().items() for partial in partial_names if partial in node["class_type"]]


def get_clone_ids(dynprompt, start_node_id, end_node_id, extra_include_partial_names):
    # get parent id chains for end node
    end_node_parent_id_chains = get_parent_id_chains(dynprompt, end_node_id)

    # find every output type node (ie nodes that have a preview)
    output_node_ids = get_all_output_node_ids(dynprompt)

    # get their parent id chains but only include chains that include the
    # start node id and not the end node id
    output_nodes_parent_id_chains = [
        chain
        for id in output_node_ids
        for chain in get_parent_id_chains(dynprompt, id)
        if start_node_id in chain and end_node_id not in chain
    ]

    all_parent_id_chains = [*end_node_parent_id_chains, *output_nodes_parent_id_chains]

    # only include chains that include any start node id
    extra_node_ids = get_ids_by_partial_names(dynprompt, extra_include_partial_names)
    start_node_ids = [start_node_id, *extra_node_ids]
    filtered = [chain for chain in all_parent_id_chains if set(start_node_ids) & set(chain)]

    # remove nodes in each chain that are "before" the start node
    trimmed = [c[:c.index(start_node_id) + 1] if start_node_id in c else c for c in filtered]

    # flatten and uniqueify
    clone_ids = list(set(item for sublist in trimmed for item in sublist))
    clone_ids.sort()
    return clone_ids

def comfyui_repeat_nodes(dynprompt, clone_ids):
    graph = GraphBuilder()

    # clone nodes
    for node_id in clone_ids:
        original_node = dynprompt.get_node(node_id)
        node = graph.node(original_node["class_type"], node_id)
        node.set_override_display_id(node_id)

    # connect cloned nodes
    for node_id in clone_ids:
        original_node = dynprompt.get_node(node_id)
        node = graph.lookup_node(node_id)
        for k, v in original_node["inputs"].items():
            if is_link(v) and v[0] in clone_ids:
                parent = graph.lookup_node(v[0])
                node.set_input(k, parent.out(v[1]))
            else:
                node.set_input(k, v)

    return graph
