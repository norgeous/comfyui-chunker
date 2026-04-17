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

def get_input_filenames():
    files = []
    input_dir = folder_paths.get_input_directory()
    for f in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, f)):
            file_parts = f.split('.')
            if len(file_parts) > 1 and (file_parts[-1].lower() in ["mp4", "mov", "webm", "png", "jpeg", "jpg", "mp3"]):
                files.append(f)
    return sorted(files)

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
    # print(name)
    lora_files = folder_paths.get_filename_list("loras")
    lora_name = None
    for lora_file in lora_files:
        if Path(lora_file).name.startswith(name) or lora_file.startswith(name):
            lora_name = lora_file
            break
    lora_path = folder_paths.get_full_path("loras", lora_name)
    # print(lora_path)
    return lora_path

# from https://github.com/yolain/ComfyUI-Easy-Use/blob/main/py/nodes/logic.py

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
            if class_type not in ["ChunkerCombine"]:
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


def comfyui_repeat_nodes(dynprompt, unique_id, start_node_id):
    graph = GraphBuilder()

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
        if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True and class_type not in ["ChunkerCombine"]:
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

    return graph

def increment_all_seeds(graph, unique_id, amt):
    # graph = GraphBuilder()
    partials = ["Sampler", "Noise"]
    prompt = graph.finalize()
    for id in prompt:
        node = prompt[id]
        class_type = node["class_type"]
        for partial in partials:
            if partial in class_type:
                real_id = id.replace(f"{unique_id}.0.0.", "")
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
