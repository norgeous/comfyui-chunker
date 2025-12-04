import torch
import math
from comfy_execution.graph_utils import GraphBuilder
from ..lib.utils import (
    count,
    log,
    # panel_image,
    # panel_mask,
    mask_to_image,
    image_to_mask,
    resize_image,
    resize_mask,
    get_input_filenames,
    force_wan_length,
    fix_total_length,
    get_this_chunk_length,
    get_audio_length,
)
from ..lib.debug_overlay import create_preview_video
from ..lib.repeatNodes import comfyuiRepeatNodes, getNodeIdsByType
from ..lib.loader import media_loader, awesome_loader, quick_combine, save_video, save_audio, get_video_info
from ..lib.loadAudio import load_audio, concat_audios

class ChunkerMediaLoader:
    @classmethod
    def INPUT_TYPES(cls):
        files = ["None", *sorted(get_input_filenames())]
        return {
            "required": {
                "images": (files, {"default": "None", "tooltip": "Images"}),
                "masks": (files, {"default": "None", "tooltip": "Masks"}),
            },
            "optional": {
                "image": (files,),
                "image_paint": (files,),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, images, masks, image, image_paint):
        # YOLO, anything goes!
        return True

    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "masks", "audio", "fps")
    OUTPUT_TOOLTIPS = (
        "images",
        "masks",
        "audio",
        "fps",
    )
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    DESCRIPTION = "ChunkerMediaLoader"

    def execute(
        self,
        images,
        masks,
        image="None",
        image_paint="None",
    ):
        out_images, out_masks, out_audio, fps = media_loader(images, masks, image, image_paint)

        ui_values = {
            "output_label_values": {
                "images": len(out_images),
                "masks": len(out_masks),
                "audio": get_audio_length(out_audio),
                "fps": fps,
            },
        }

        return {
            "ui": {"values": [ui_values]},
            "result": (
                out_images,
                out_masks,
                out_audio,
                fps,
            ),
        }

class Chunker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["None", "Wan21", "Wan22"], {"tooltip": "Force chunk lengths to match Wan's format 4n+1. 16fps for Wan21, 24fps for Wan22"}),
                "chunk_length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 1, "tooltip": "Count of images in each chunk"}),
                "chunk_overlap": ("INT", {"default": 4, "min": 0, "max": 4096, "step": 1, "tooltip": "Count of images to overlap between chunks"}),
                "total_length": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "tooltip": "Minimum count of images in the final output. 0 to use the images length"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "images"}),
                "masks": ("MASK", {"tooltip": "masks"}),
                "audio": ("AUDIO", {"tooltip": "audio"}),
                "fps": ("FLOAT", {"forceInput": True, "tooltip": "fps"}),
                "store": ("*",), # hidden by js
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("CHUNKER_DATA", "IMAGE", "MASK", "AUDIO", "INT", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("chunker_data", "images", "masks", "audio", "width", "height", "chunk_length", "chunk_overlap", "total_length", "chunk_count", "index")
    OUTPUT_TOOLTIPS = (
        "Connect \"chunker_data\" to the \"ChunkerCombine\" node",
        "Chunk of images",
        "Chunk of masks",
        "Chunk of audio",
        "Width of images",
        "Height of images",
        "Count of images in each chunk",
        "Count of images to overlap between each chunk",
        "Total length of output images",
        "Count of chunks",
        "The current itteration index, ie; 0, 1, 2, ...",
    )
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    DESCRIPTION = "Chunker"

    def execute(
        self,
        mode,
        chunk_length,
        chunk_overlap,
        total_length,
        images=None,
        masks=None,
        audio=None,
        fps=None,
        store=None,
        unique_id=None,
    ):
        s = store if store is not None else {
            "index": 0,
            "images_last_chunk_path": None,
            "masks_last_chunk_path": None,
            "audio_last_chunk_path": None,
        }

        if fps is None and mode == "Wan21": fps = 16
        if fps is None and mode == "Wan22": fps = 24
        if fps is None: fps = 30

        if total_length == 0:
            total_length = max(
                len(images) if images is not None else 0,
                len(masks) if images is not None else 0,
            )

        if mode == "Wan21" or mode == "Wan22":
            chunk_length = force_wan_length(chunk_length)
            total_length = fix_total_length(total_length, chunk_length, chunk_overlap)

        this_chunk_length = get_this_chunk_length(s["index"], chunk_length, chunk_overlap, total_length)

        w = None
        h = None

        start = (s["index"] * (chunk_length - chunk_overlap))
        end = start + chunk_length
        chunk_count = math.ceil((total_length - chunk_overlap) / (chunk_length - chunk_overlap))

        out_images = []
        out_masks = []
        out_audio = []

        # get the images overlap from last chunk
        if s["images_last_chunk_path"] is not None:
            images_overlap = awesome_loader(s["images_last_chunk_path"], start=-chunk_overlap)[0]
            w = images_overlap.shape[2]
            h = images_overlap.shape[1]
            out_images.append(images_overlap)

        # get the masks overlap from last chunk
        if s["masks_last_chunk_path"] is not None:
            imasks_overlap = awesome_loader(s["masks_last_chunk_path"], start=-chunk_overlap)[0]
            masks_overlap = image_to_mask(imasks_overlap)
            out_masks.append(masks_overlap)

        samples_per_frame = math.floor(audio["sample_rate"] / fps)
        astart = (start + count(out_images)) * samples_per_frame
        aend = end * samples_per_frame

        # get the audio overlap from last chunk
        if s["audio_last_chunk_path"] is not None:
            print('loading mp3', s["audio_last_chunk_path"])
            audio_overlap = awesome_loader(s["audio_last_chunk_path"], start=-chunk_overlap)[0]
            out_audio.append(audio_overlap)

        if (mode == "Wan21" or mode == "Wan22") and (count(out_images) > count(out_masks)):
            black_panel = torch.full((1, w, h), 0) # panel_mask(w, h, 0)
            out_masks.append(torch.cat([black_panel] * (count(out_images) - count(out_masks)))) # add same amount of black masks to masks

        if audio is not None:
            out_audio.append({
                "waveform": audio["waveform"][:,:,astart:aend],
                "sample_rate": audio["sample_rate"],
            })

        if images is not None:
            if w is None: w = images.shape[2]
            if h is None: h = images.shape[1]
            out_images.append(images[start + count(out_images):end])

        if masks is not None:
            out_masks.append(masks[start + count(out_masks):end])

        if w is None: w = 512
        if h is None: h = 512

        grey_panel = torch.full((1, w, h, 3), 0.5) # panel_image(w, h, 127, 127, 127)
        white_panel = torch.full((1, w, h), 1) # panel_mask(w, h, 255)
        black_panel = torch.full((1, w, h), 0) # panel_mask(w, h, 0)

        # do some stuff for Wan
        if mode == "Wan21" or mode == "Wan22":
            # grey_panel = torch.full((1, w, h, 3), 0.5) # panel_image(w, h, 128, 128, 128)
            # white_panel = torch.full((1, w, h), 1) # panel_mask(w, h, 255)

            # if not enough images, invent some blank (grey) ones (for t2v)
            if count(out_images) < this_chunk_length: out_images.append(torch.cat([grey_panel] * (this_chunk_length - count(out_images))))

            # if not enough masks, invent some blank (white) ones (for t2v)
            if count(out_masks) < this_chunk_length: out_masks.append(torch.cat([white_panel] * (this_chunk_length - count(out_masks))))

        out_images_torch = None
        if len(out_images) > 0:
            out_images_resized = list(map(lambda tensor: resize_image(tensor, w, h), out_images))
            out_images_torch = torch.cat(out_images_resized)
            assert len(out_images_torch.shape) == 4, f"images are not rank 4 {out_images_torch.shape}"

        out_masks_torch = None
        if len(out_masks) > 0:
            out_masks_resized = list(map(lambda tensor: resize_mask(tensor, w, h), out_masks))
            out_masks_torch = torch.cat(out_masks_resized)
            assert len(out_masks_torch.shape) == 3, f"masks are not rank 3 {out_masks_torch.shape}"

        out_audio_dict = None
        if len(out_audio) > 0:
            out_audio_dict = concat_audios(out_audio)

        c = {
            "mode": mode,
            "chunk_length": chunk_length,
            "chunk_overlap": chunk_overlap,
            "total_length": total_length,
            "chunk_count": chunk_count,
        }

        chunker_data = {
            "start_node_id": unique_id,
            "index": s["index"],
            "chunker_config": c,
            "fps": fps,
        }

        ui_values = {
            "input_label_values": {
                "images": len(images) if images is not None else 0,
                "masks": len(masks) if masks is not None else 0,
                "audio": get_audio_length(audio),
                "fps": fps,
            },
            "output_label_values": {
                "images": count(out_images),
                "masks": count(out_masks),
                "audio": get_audio_length(out_audio_dict), # TODO: chop up audio from input video or overlap
                "width": w,
                "height": h,
                "chunk_length": max(count(out_images), count(out_masks)),
                "chunk_overlap": chunk_overlap,
                "total_length": total_length,
                "chunk_count": chunk_count,
                "index": s["index"],
            },
        }

        log(f"Starting chunk {s["index"] + 1} of {c["chunk_count"]}...")

        return {
            "ui": {"values": [ui_values]},
            "result": (
                chunker_data,
                out_images_torch,
                out_masks_torch,
                out_audio_dict,
                w,
                h,
                max(count(out_images), count(out_masks)),
                chunk_overlap,
                total_length,
                chunk_count,
                s["index"],
            ),
        }

class ChunkerVACEToFirstLast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_vision": ("CLIP_VISION",),
                "images": ("IMAGE", {"tooltip": "VACE image sequence"}),
                "crop": (["none", "center"],)
            },
        }

    RETURN_TYPES = ("CLIP_VISION_OUTPUT", "CLIP_VISION_OUTPUT", "IMAGE", "IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("clip_vision_start_image", "clip_vision_end_image", "start_image", "end_image", "width", "height", "length")
    OUTPUT_TOOLTIPS = (
        "Start clip vison or None",
        "End clip vision or None",
        "Start image or None",
        "End image or None",
        "width",
        "height",
        "length",
    )
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    DESCRIPTION = "ChunkerVACEToFirstLast"

    def execute(self, clip_vision, images, crop):
        length = images.shape[0]
        h = images.shape[1]
        w = images.shape[2]

        grey_panel = torch.full((1, w, h, 3), 0.5)

        clip_vision_start_image = None
        start_image = images[0].unsqueeze(0)
        if round(torch.mean(start_image).item(), 4) == round(127 / 255, 4): # detect grey panels
            start_image = None
        else:
            clip_vision_start_image = clip_vision.encode_image(start_image, crop=False if crop == "center" else True)

        clip_vision_end_image = None
        end_image = images[length - 1].unsqueeze(0)
        if round(torch.mean(end_image).item(), 4) == round(127 / 255, 4): # detect grey panels
            end_image = None
        else:
            clip_vision_end_image = clip_vision.encode_image(end_image, crop=False if crop == "center" else True)

        ui_values = {
            "input_label_values": {
                "images": len(images) if images is not None else 0,
            },
            "output_label_values": {
                "clip_vision_start_image": 1 if clip_vision_start_image is not None else 0,
                "clip_vision_end_image": 1 if clip_vision_end_image is not None else 0,
                "start_image": 1 if start_image is not None else 0,
                "end_image": 1 if end_image is not None else 0,
                "width": w,
                "height": h,
                "length": length,
            },
        }

        return {
            "ui": {"values": [ui_values]},
            "result": (
                clip_vision_start_image,
                clip_vision_end_image,
                start_image,
                end_image,
                w,
                h,
                length,
            ),
        }

class ChunkerCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunker_data": ("CHUNKER_DATA", {"tooltip": "Connect chunker_data from Chunker node to here"}),
                "show_debug": ("BOOLEAN", {"default": True, "tooltip": "Show debug overlay in preview"}),
                "select_overlaps_from": (["this_chunk", "previous_chunk"], {"default": "this_chunk", "tooltip": "TODO"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Processed chunk of images"}),
                "masks": ("MASK", {"tooltip": "Processed chunk of masks"}),
                "audio": ("AUDIO", {"tooltip": "Processed chunk of audio"}),
                "store": ("*",), # hidden by js
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "FLOAT")
    RETURN_NAMES = ("images", "masks", "audio", "fps")
    OUTPUT_TOOLTIPS = (
        "Combined images from all chunks",
        "Combined masks from all chunks",
        "Audio from images input of Chunker",
        "FPS",
    )
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    DESCRIPTION = "ChunkerCombine"
    OUTPUT_NODE = True

    def execute(
        self,
        chunker_data,
        show_debug,
        select_overlaps_from,
        images=None,
        masks=None,
        audio=None,
        store=None,
        dynprompt=None,
        unique_id=None,
    ):
        if images is None and masks is None:
            raise Exception("Please provide images OR masks")

        d = chunker_data
        c = d["chunker_config"]
        s = store if store is not None else {
            "image_chunks": [],
            "mask_chunks": [],
            "audio_chunks": [],
            "preview_chunks": [],
        }

        # figure out if we have completed all chunks
        is_done = d["index"] + 1 >= c["chunk_count"]

        # save new image chunk to a new file
        if images is not None:
            images_full_path = save_video(images, d["fps"], "video/chunker/tmp/chunk/image_chunk", audio)[0]
            s["image_chunks"].append(images_full_path)

        # save new mask chunk to a new file
        if masks is not None:
            masks_full_path = save_video(mask_to_image(masks), d["fps"], "video/chunker/tmp/chunk/mask_chunk")[0]
            s["mask_chunks"].append(masks_full_path)

        # save new audio chunk to a new file
        if audio is not None:
            audio_full_path = save_audio(audio, "video/chunker/tmp/chunk/audio_chunk")[0]
            print(audio_full_path)
            # s["audio_chunks"].append(audio)
            s["audio_chunks"].append(audio_full_path)

        # create preview from inputs
        preview = create_preview_video(images, masks, show_debug, d, c)

        # save new preview chunk to a new file
        preview_full_path = save_video(preview, d["fps"], "video/chunker/tmp/chunk/preview_chunk", audio)[0]
        s["preview_chunks"].append(preview_full_path)

        #print("these should have audio", s["image_chunks"], s["preview_chunks"])

        # combine all preview chunks to a new file, excluding the overlaps
        filename_prefix = "video/chunker/tmp/chunks/preview_chunks" if not is_done else "video/chunker/tmp/chunks/preview_complete"
        all_preview_frontend_data = quick_combine(s["preview_chunks"], c["chunk_overlap"], select_overlaps_from, filename_prefix)[1]

        # if no more chunks needed, return early
        if is_done:
            # load all image chunks as tensor, excluding the overlaps
            out_images_torch = None
            if len(s["image_chunks"]) > 0:
                log("[debug] Combine -> combine all images...", end="")
                all_images_video_path = quick_combine(s["image_chunks"], c["chunk_overlap"], select_overlaps_from, "video/chunker/images")[0]
                print("done")
                log("[debug] Combine -> load all images as tensor...", end="")
                out_images_torch = awesome_loader(all_images_video_path)[0]
                print("done")

            # load all mask chunks as tensor, excluding the overlaps
            out_masks_torch = None
            if len(s["mask_chunks"]) > 0:
                log("[debug] Combine -> combine all masks...", end="")
                all_masks_video_path = quick_combine(s["mask_chunks"], c["chunk_overlap"], select_overlaps_from, "video/chunker/masks")[0]
                print("done")
                log("[debug] Combine -> load all masks as tensor...", end="")
                out_masks_torch = awesome_loader(all_masks_video_path)[0]
                print("done")

            # load all audio chunks as tensor
            out_audio_dict = None
            if len(s["audio_chunks"]) > 0:
                out_audio_dict = concat_audios(s["audio_chunks"])
                # load_audio(d["audio"]) if d["audio"] is not None else None,

            ui_values = {
                "input_label_values": {
                    "images": len(images) if images is not None else 0,
                    "masks": len(masks) if masks is not None else 0,
                    "audio": get_audio_length(audio),
                },
                "output_label_values": {
                    "images": len(out_images_torch) if out_images_torch is not None else 0,
                    "masks": len(out_masks_torch) if out_masks_torch is not None else 0,
                    "audio": get_audio_length(out_audio_dict),
                    "fps": f"{d["fps"]:.2f}",
                },
                "index": d["index"],
                "chunk_count": c["chunk_count"],
                "video_path": all_preview_frontend_data,
            }

            log(f"Finished all chunks {d["index"] + 1} of {c["chunk_count"]}!")

            return {
                "ui": {"values": [ui_values]},
                "result":(
                    out_images_torch,
                    image_to_mask(out_masks_torch),
                    out_audio_dict,
                    d["fps"],
                )
            }

        # clone all the nodes between Chunker and ChunkerCombine
        graph = GraphBuilder()
        comfyuiRepeatNodes(dynprompt, graph, unique_id, d["start_node_id"])

        # update the store in the new_chunker
        new_chunker = graph.lookup_node(d["start_node_id"])
        new_chunker.set_input("store", {
            "index": d["index"] + 1,
            "images_last_chunk_path": s["image_chunks"][-1] if len(s["image_chunks"]) > 0 else None, # filename of last image chunk saved
            "masks_last_chunk_path": s["mask_chunks"][-1] if len(s["mask_chunks"]) > 0 else None, # filename of last mask chunk saved
            "audio_last_chunk_path": s["audio_chunks"][-1] if len(s["audio_chunks"]) > 0 else None, # filename of last audio chunk saved
        })

        # increment seeds in cloned KSamplers, to prevent same motion in each chunk (for Wan)
        ids = getNodeIdsByType(graph.finalize(), "KSampler")
        for id in ids:
            real_id = id.replace(f"{unique_id}.0.0.", "")
            node = graph.lookup_node(real_id)
            seed = node.get_input("seed")
            node.set_input("seed", seed + d["index"] + 1)

        # increment seeds in cloned KSamplersAdvanced, to prevent same motion in each chunk (for Wan)
        ids = getNodeIdsByType(graph.finalize(), "KSamplerAdvanced")
        for id in ids:
            real_id = id.replace(f"{unique_id}.0.0.", "")
            node = graph.lookup_node(real_id)
            seed = node.get_input("noise_seed")
            node.set_input("noise_seed", seed + d["index"] + 1)

        # increment seeds in cloned mmaudio
        #ids = getNodeIdsByType(graph.finalize(), "MMAudioSampler")
        #for id in ids:
        #    real_id = id.replace(f"{unique_id}.0.0.", "")
        #    node = graph.lookup_node(real_id)
        #    seed = node.get_input("seed")
        #    node.set_input("seed", seed + d["index"] + 1)

        # update the store in the new_combine (this node)
        new_combine = graph.lookup_node("Recurse")
        new_combine.set_input("store", s)

        ui_values = {
            "input_label_values": {
                "images": len(images) if images is not None else 0,
                "masks": len(masks) if masks is not None else 0,
                "audio": get_audio_length(audio),
            },
            "output_label_values": {
                "images": None,
                "masks": None,
                "audio": None,
                "fps": None,
            },
            "index": d["index"],
            "chunk_count": c["chunk_count"],
            "video_path": all_preview_frontend_data,
        }

        log(f"Finished chunk {d["index"] + 1} of {c["chunk_count"]}")

        return {
            "ui": {"values": [ui_values]},
            "result": (
                new_combine.out(0),
                new_combine.out(1),
                new_combine.out(2),
                new_combine.out(3),
            ),
            "expand": graph.finalize(),
        }
