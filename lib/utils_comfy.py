from pathlib import Path
import os
import folder_paths
from comfy_execution.graph_utils import GraphBuilder, is_link
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS
from functools import reduce
from comfy_extras.nodes_audio import match_audio_sample_rates
import torch

def concat_audio(audio1, audio2):
    waveform_1 = audio1["waveform"]
    waveform_2 = audio2["waveform"]
    sample_rate_1 = audio1["sample_rate"]
    sample_rate_2 = audio2["sample_rate"]
    waveform_1, waveform_2, output_sample_rate = match_audio_sample_rates(waveform_1, sample_rate_1, waveform_2, sample_rate_2)
    concatenated_audio = torch.cat((waveform_1, waveform_2), dim=2)
    return {
        "waveform": concatenated_audio,
        "sample_rate": output_sample_rate,
    }

def concat_audios(audios):
    return reduce(lambda a, b: concat_audio(a, b), audios)

def get_next_save_path(filename_prefix, extension):
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix,
        folder_paths.get_output_directory(),
    )
    file = f"{filename}_{counter:05}_.{extension}"
    full_path = os.path.join(full_output_folder, file)
    return (
        full_path,
        {
            "filename": file,
            "subfolder": subfolder,
            "type": "output",
        },
    )

def find_lora_full_path(name):
    lora_files = folder_paths.get_filename_list("loras")
    lora_name = None
    for lora_file in lora_files:
        if Path(lora_file).name.startswith(name) or lora_file.startswith(name):
            lora_name = lora_file
            break
    lora_path = folder_paths.get_full_path("loras", lora_name)
    return lora_path

def get_parent_ids(dynprompt, node_id):
    node_info = dynprompt.get_node(node_id)
    if "inputs" not in node_info: return []
    parent_ids = []
    for k, v in node_info["inputs"].items():
        if is_link(v):
            parent_id = v[0]
            grandparent_ids = get_parent_ids(dynprompt, parent_id)
            parent_ids.append([parent_id, *grandparent_ids])
    return parent_ids

def extract_chains(node, chain=[]):
    if isinstance(node, str): return [chain + [node]]
    node_val = node[0]
    new_chain = chain + [node_val]
    children = node[1:]
    if not children: return [new_chain]
    paths = []
    for child in children: paths.extend(extract_chains(child, new_chain))
    return paths

def get_parent_id_chains(dynprompt, node_id):
    return extract_chains([node_id, *get_parent_ids(dynprompt, node_id)])

def get_ids_all_output_nodes(dynprompt):
    prompt = dynprompt.get_original_prompt()
    return [id for id, info in prompt.items() if getattr(ALL_NODE_CLASS_MAPPINGS.get(info.get("class_type")), 'OUTPUT_NODE', False) is True]

def get_ids_by_partial_names(dynprompt, partial_names):
    prompt = dynprompt.get_original_prompt()
    return [id for partial in partial_names for id, info in prompt.items() if partial in info["class_type"]]

def comfyui_repeat_nodes(dynprompt, end_node_id, start_node_id):
    from pprint import pprint
    print()

    print(f"comfyui_repeat_nodes ({start_node_id},{end_node_id})")

    # get parent id chains for end node
    end_node_parent_id_chains = get_parent_id_chains(dynprompt, end_node_id)

    # find every output type node (ie nodes that have a preview)
    output_node_ids = get_ids_all_output_nodes(dynprompt)

    # get their parent id chains but only include chains that include the start node id
    output_nodes_parent_id_chains = [chain for id in output_node_ids for chain in get_parent_id_chains(dynprompt, id) if start_node_id in chain]

    pprint(output_nodes_parent_id_chains)

    all_parent_id_chains = [*end_node_parent_id_chains,*output_nodes_parent_id_chains]
    extra_node_ids = get_ids_by_partial_names(dynprompt, ["Noise"])
    start_node_ids = [start_node_id, *extra_node_ids]
    
    # only include chains that include any start node id
    filtered = [chain for chain in all_parent_id_chains if set(start_node_ids) & set(chain)]
    
    # remove nodes in each chain that are "before" the start node
    trimmed = [c[:c.index(start_node_id)+1] if start_node_id in c else c for c in filtered]
    
    # flatten and uniqueify
    clone_ids = list(set(item for sublist in trimmed for item in sublist))
    
    pprint(clone_ids)

    graph = GraphBuilder()
    for node_id in clone_ids:
        original_node = dynprompt.get_node(node_id)
        node = graph.node(original_node["class_type"], "Recurse" if node_id == end_node_id else node_id)
        node.set_override_display_id(node_id)
    for node_id in clone_ids:
        original_node = dynprompt.get_node(node_id)
        node = graph.lookup_node("Recurse" if node_id == end_node_id else node_id)
        for k, v in original_node["inputs"].items():
            if is_link(v) and v[0] in clone_ids:
                parent = graph.lookup_node(v[0])
                node.set_input(k, parent.out(v[1]))
            else:
                node.set_input(k, v)

    return graph

def increment_all_seeds(graph, end_node_id, amt):
    partials = ["Sampler", "Noise"]
    prompt = graph.finalize()
    for id in prompt:
        node = prompt[id]
        class_type = node["class_type"]
        for partial in partials:
            if partial in class_type:
                real_id = id.replace(f"{end_node_id}.0.0.", "")
                node = graph.lookup_node(real_id)

                # if node has a disconnected seed input
                seed = node.get_input("seed")
                if isinstance(seed, int):
                    new_seed = seed + amt
                    node.set_input("seed", new_seed)

                # if node has a disconnected noise_seed input
                noise_seed = node.get_input("noise_seed")
                if isinstance(noise_seed, int):
                    new_noise_seed = noise_seed + amt
                    node.set_input("noise_seed", new_noise_seed)

                break
