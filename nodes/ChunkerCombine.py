import comfy.utils
from comfy_api.latest import io, VideoFromFile
from ..lib.utils import log
from ..lib.av_save import av_save, Profile
from ..lib.av_combine import av_combine, BlendMode
from ..lib.utils_tensor import resize_mask
from ..lib.create_preview_video import create_preview_video
from ..lib.utils_comfy_repeat_nodes import get_clone_ids, comfyui_repeat_nodes, get_ids_by_partial_names, get_ids_by_partial_names_in_graph
from ..lib.utils_format import format_images, format_masks, format_audio, format_fps, format_milliseconds, format_video
from ..lib.utils_performance import get_ts
from ..lib.calculate_progress_bar import calculate_progress_bar
from ..lib.execution_monitor import get_execution_start_time


def _detect_connected_outputs(prompt, dynprompt, node_id: str) -> set[int]:
    root_id = dynprompt.get_display_node_id(node_id)
    connected = set()
    for nid, info in prompt.items():
        if nid == root_id:
            continue
        for val in info.get("inputs", {}).values():
            if isinstance(val, list) and len(val) == 2 and val[0] == root_id:
                connected.add(int(val[1]))
    return connected


def collect_seed_info(dynprompt, clone_ids: list[str]) -> str:
    seed_node_ids = [id for id in get_ids_by_partial_names(dynprompt, ["Sampler", "Noise"]) if id in clone_ids]
    lines = []
    for id in seed_node_ids:
        node_info = dynprompt.get_node(id)
        class_type = node_info["class_type"]
        original_id = dynprompt.get_display_node_id(id)
        seed = node_info["inputs"].get("seed")
        noise_seed = node_info["inputs"].get("noise_seed")
        if seed is not None and isinstance(seed, int):
            lines.append(f"{class_type}#{original_id}: {seed}")
        if noise_seed is not None and isinstance(noise_seed, int):
            lines.append(f"{class_type}#{original_id}: {noise_seed}")
    return "\n".join(lines)


class ChunkerCombine(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ChunkerCombine",
            display_name="\U0001F36B Combine",
            category="chunker",
            description=(
                "Combine sequential chunks back into single av tensors. Each chunk is saved to a mp4 file and recombined excluding the overlap frames.\n"
                "\n"
                "Note: You may need a (very) large pagefile or swap configured."
            ),
            inputs=[
                io.Custom("CHUNKER_DATA").Input(
                    "chunker_data",
                    tooltip="Connect chunker_data from ChunkerRepeat node to here",
                ),
                io.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Processed chunk of images",
                ),
                io.Mask.Input(
                    "masks",
                    optional=True,
                    tooltip="Processed chunk of masks",
                ),
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Processed chunk of audio",
                ),
                io.Combo.Input(
                    "overlap_blend_mode",
                    options=list(map(lambda member: member.value, BlendMode)),
                    default=BlendMode.LINEAR.value,
                    tooltip="When chunk_overlap is more than zero this setting determines how images and audio are blended",
                ),
                io.Boolean.Input(
                    "increment_seeds",
                    tooltip=("Increment \"seed\" or \"noise_seed\" inputs in repeated nodes that have \"Sampler\" or \"Noise\" within the node class name"),
                    default=True,
                ),
                io.Custom("*").Input(
                    "store",
                    optional=True,
                ),
            ],
            outputs=[
                io.Video.Output(
                    "video",
                    tooltip="Combined video file",
                ),
                io.Image.Output(
                    "images",
                    tooltip="Combined images from all chunks",
                ),
                io.Mask.Output(
                    "masks",
                    tooltip="Combined masks from all chunks",
                ),
                io.Audio.Output(
                    "audio",
                    tooltip="Combined audio from all chunks",
                ),
                io.Float.Output(
                    "fps",
                    tooltip="FPS",
                ),
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.dynprompt, io.Hidden.prompt],
            is_output_node=True,
            enable_expand=True,
        )

    @classmethod
    def execute(
        self,
        chunker_data,
        overlap_blend_mode,
        increment_seeds,
        images=None,
        masks=None,
        audio=None,
        store=None,
    ):
        if images is None and masks is None and audio is None:
            raise ValueError("At least one of images, masks, or audio must be provided.")

        d = chunker_data
        c = d["chunker_config"]
        s = store if store is not None else {
            "chunks": [],
            "previews": [],
            "ts_chunk_ends": [],
        }

        pbar = comfy.utils.ProgressBar(0, node_id=self.hidden.dynprompt.get_display_node_id(self.hidden.unique_id))

        # lanczos resize masks to match images size
        if images is not None and masks is not None:
            masks = resize_mask(masks, images.shape[2], images.shape[1])

        # Save images, masks and audio to lossless file
        ts = get_ts()
        log("Save chunk...", end="")
        chunk_path, _ = av_save(
            images=images,
            masks=masks,
            audio=audio,
            fps=d["fps"],
            filename_prefix="chunker-chunk",
        )
        s["chunks"].append(chunk_path)
        print(f"done ({format_milliseconds(get_ts() - ts)})")

        # Identify nodes to repeat and collect seed info from prompt
        clone_ids = get_clone_ids(self.hidden.dynprompt, d["start_node_id"], self.hidden.unique_id, ["Noise"] if increment_seeds else [])
        seed_ids = get_clone_ids(self.hidden.dynprompt, d["start_node_id"], self.hidden.unique_id, ["Noise"])
        seed_info = collect_seed_info(self.hidden.dynprompt, seed_ids)

        # Make preview from inputs
        ts = get_ts()
        log("Make preview...", end="")
        preview_images, preview_masks, preview_audio, preview_fps = create_preview_video(images, masks, audio, d, c, overlap_blend_mode, seed_info)
        print(f"done ({format_milliseconds(get_ts() - ts)})")

        # Save preview to web file
        ts = get_ts()
        log("Save preview...", end="")
        preview_path, _ = av_save(
            images=preview_images,
            masks=preview_masks,
            audio=preview_audio,
            fps=d["fps"],
            filename_prefix="chunker-preview",
            profile=Profile.WEBRGB if masks is None else Profile.WEBRGBA,
        )
        s["previews"].append(preview_path)
        print(f"done ({format_milliseconds(get_ts() - ts)})")

        # Combine all preview chunks to a new file, blending the overlaps
        ts = get_ts()
        log("Combine all previews...", end="")
        _, all_preview_frontend_data, _, _, _ = av_combine(
            inputs=[*s["previews"][:-1], (preview_images, preview_masks, preview_audio, preview_fps)],
            filename_prefix="chunker-preview-all",
            overlap_frame_count=c["chunk_overlap"],
            video_blend_mode=BlendMode(overlap_blend_mode),
            audio_blend_mode=BlendMode(overlap_blend_mode),
            profile=Profile.WEBRGB if masks is None else Profile.WEBRGBA,
        )
        print(f"done ({format_milliseconds(get_ts() - ts)})")

        # figure out if we have completed all chunks
        is_done = d["index"] + 1 >= c["chunk_count"]

        # if no more chunks needed, return early
        if is_done:
            connected = _detect_connected_outputs(self.hidden.prompt, self.hidden.dynprompt, self.hidden.unique_id)

            ts = get_ts()
            log("Combine all chunks...", end="")
            out_video_path, _, out_images_torch, out_masks_torch, out_audio_dict = av_combine(
                inputs=s["chunks"],
                filename_prefix="chunker-chunk-all",
                overlap_frame_count=c["chunk_overlap"],
                video_blend_mode=BlendMode(overlap_blend_mode),
                audio_blend_mode=BlendMode(overlap_blend_mode),
                need_images=1 in connected,
                need_masks=2 in connected,
                need_audio=3 in connected,
            )
            print(f"done ({format_milliseconds(get_ts() - ts)})")
            out_video = VideoFromFile(out_video_path)

            s["ts_chunk_ends"] = [
                *s["ts_chunk_ends"],
                get_ts(),
            ]

            ui_values = {
                "input_label_values": {
                    "images": format_images(images),
                    "masks": format_masks(masks),
                    "audio": format_audio(audio),
                },
                "output_label_values": {
                    "video": format_video(out_video),
                    "images": format_images(out_images_torch),
                    "masks": format_masks(out_masks_torch),
                    "audio": format_audio(out_audio_dict),
                    "fps": format_fps(d["fps"]),
                },
                "bar": calculate_progress_bar(get_execution_start_time(), d["ts_chunk_starts"], s["ts_chunk_ends"], c["chunk_count"], d["chunk_lengths"]),
                "video_path": all_preview_frontend_data,
            }

            pbar.update_absolute(d["index"] + 1, c["chunk_count"])

            log(f"ChunkerCombine#{self.hidden.unique_id}: Finished all chunks {d['index'] + 1} of {c['chunk_count']}!")

            return {
                "ui": {"values": [ui_values]},
                "result": (
                    out_video,
                    out_images_torch,
                    out_masks_torch,
                    out_audio_dict,
                    float(d["fps"]),
                )
            }

        # Clone all nodes between ChunkerRepeat and ChunkerCombine
        log(f"ChunkerCombine#{self.hidden.unique_id}: cloning {len(clone_ids)} nodes (max nesting depth {max(id.count('.') for id in clone_ids)})")
        for id in clone_ids: log(f'Repeating node {self.hidden.dynprompt.get_node(id)["class_type"]}#{self.hidden.dynprompt.get_display_node_id(id)}')
        graph = comfyui_repeat_nodes(self.hidden.dynprompt, clone_ids)

        # Increment seed in cloned nodes with "Sampler" or "Noise" in the node type name, such as;
        # - KSampler
        # - KSamplerAdvanced
        # - RandomNoise (used by SamplerCustomAdvanced)
        # - ClownsharKSampler
        # this is to prevent same motion in each chunk (when using Wan or LTX)
        seed_nodes = get_ids_by_partial_names_in_graph(graph, ["Sampler", "Noise"])
        for id in seed_nodes:
            real_id = id.replace(f"{self.hidden.unique_id}.0.0.", "")
            original_id = self.hidden.dynprompt.get_display_node_id(real_id)
            node = graph.lookup_node(real_id)
            class_type = self.hidden.dynprompt.get_node(original_id)['class_type']

            seed = node.get_input("seed")
            if isinstance(seed, int):
                new_seed = seed + 1
                log(f"Increment seed in {class_type}#{original_id}; {seed} -> {new_seed}")
                node.set_input("seed", new_seed)

            noise_seed = node.get_input("noise_seed")
            if isinstance(noise_seed, int):
                new_noise_seed = noise_seed + 1
                log(f"Increment noise_seed in {class_type}#{original_id}; {noise_seed} -> {new_noise_seed}")
                node.set_input("noise_seed", new_noise_seed)

        # update the store in the cloned ChunkerRepeat
        new_repeat = graph.lookup_node(d["start_node_id"])
        new_repeat.set_input("store", {
            "index": d["index"] + 1,
            "last_chunk_path": s["chunks"][-1],
            "ts_chunk_starts": d["ts_chunk_starts"],
        })

        # update the store in the cloned ChunkerCombine (copy of this node)
        new_combine = graph.lookup_node(self.hidden.unique_id)
        s["ts_chunk_ends"] = [
            *s["ts_chunk_ends"],
            get_ts(),
        ]
        new_combine.set_input("store", s)

        ui_values = {
            "input_label_values": {
                "images": format_images(images),
                "masks": format_masks(masks),
                "audio": format_audio(audio),
            },
            "output_label_values": {
                "video": None,
                "images": None,
                "masks": None,
                "audio": None,
                "fps": None,
            },
            "bar": calculate_progress_bar(get_execution_start_time(), d["ts_chunk_starts"], s["ts_chunk_ends"], c["chunk_count"], d["chunk_lengths"]),
            "video_path": all_preview_frontend_data,
        }

        pbar.update_absolute(d["index"] + 1, c["chunk_count"])

        log(f"ChunkerCombine#{self.hidden.unique_id}: Finished chunk {d['index'] + 1} of {c['chunk_count']}")

        return io.NodeOutput(
            new_combine.out(0), # video
            new_combine.out(1), # images
            new_combine.out(2), # masks
            new_combine.out(3), # audio
            new_combine.out(4), # fps
            ui={"values": [ui_values]},
            expand=graph.finalize(),
        )
