import os
import folder_paths
import torch
import math
from comfy_execution.graph_utils import GraphBuilder
from .utils import (
    count,
    log,
    panel_image,
    panel_mask,
    mask_to_image,
    image_to_mask,
    resize_image,
    resize_mask,
    get_input_filenames,
    force_wan_length,
    fix_total_length,
    get_this_chunk_length,
    obscure_image,
    expand_image,
)
from .debug_overlay import create_preview_video
from .repeatNodes import comfyuiRepeatNodes, getNodeIdsByType
from .loader import media_loader, awesome_loader, quick_combine, save_video, get_video_info
from .loadAudio import load_audio, concat_audios

#def parse_config_paths(chunk_config):
#    images = chunk_config["images"]
#    masks = chunk_config["masks"]
#    image = chunk_config["image"]
#    image_paint = chunk_config["image_paint"]
#    if images == "None": images = None
#    if masks == "None": masks = None
#    if image == "None": image = None
#    if image_paint == "None": image_paint = None
#    if images is not None: images = folder_paths.get_annotated_filepath(images)
#    if masks is not None: masks = folder_paths.get_annotated_filepath(masks)
#    if image is not None: image = folder_paths.get_annotated_filepath(image)
#    if image_paint is not None: image_paint = folder_paths.get_annotated_filepath(image_paint)
#    return (
#        images,
#        masks,
#        image,
#        image_paint,
#    )

def get_audio_length(audio):
    if audio is None: return "0s"
    return f"{audio["waveform"].shape[2] / audio["sample_rate"]:.6f}s"


















class ChunkerMediaLoaderOld:
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

    RETURN_TYPES = ("PATHS", "IMAGE", "MASK", "AUDIO")
    RETURN_NAMES = ("paths", "images", "masks", "audio")
    OUTPUT_TOOLTIPS = (
        "paths",
        "images",
        "masks",
        "audio",
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
        paths = {
            "images": images,
            "masks": masks,
            "image": image,
            "image_paint": image_paint,
        }
        return (
            paths,
            None, # TODO: images
            None, # TODO: masks
            None, # TODO: audio
        )


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

class ChunkerChunkPlacement:
    @classmethod
    def INPUT_TYPES(cls):
        files = ["None", *sorted(get_input_filenames())]
        return {
            "required": {
                "include_in": (["specified_chunk_only", "every_nth_chunk", "every_chunk"], {}),
                "chunk": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "Which chunk these settings affect"}),
                "frame": (["start", "end", "every"], {"default": "start", "tooltip": "Frames within the chunk that the image will appear"}),
                #"obscure": ("BOOLEAN", {"default": False, "tooltip": "Fill grey in the image inside the masked area"}),
            },
            "optional": {
                "chunk_configs": ("CHUNK_CONFIGS", {"tooltip": "Previous chunk_config for chaining"}),
                "paths": ("PATHS",),
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("CHUNK_CONFIGS",)
    RETURN_NAMES = ("chunk_configs",)
    OUTPUT_TOOLTIPS = (
        "Chunk config",
    )
    FUNCTION = "execute"
    CATEGORY = "Chunker"
    DESCRIPTION = "ChunkerChunkPlacement"

    def execute(
        self,
        include_in,
        chunk,
        frame,
        #obscure,
        chunk_configs=None,
        paths=None,
        images=None,
        masks=None,
        audio=None,
    ):
        if paths == None and images == None and masks == None and audio == None: raise Exception("Please connect an input, one of; paths OR images OR masks OR audio")
        chunk_config = {
            "include_in": include_in,
            "chunk": chunk,
            "frame": frame,
            #"obscure": obscure,
        }
        if paths is not None:
            chunk_config["paths"] = paths
            chunk_config["images"] = None
            chunk_config["masks"] = None
            chunk_config["audio"] = None
        else:
            chunk_config["paths"] = None
            chunk_config["images"] = images
            chunk_config["masks"] = masks
            chunk_config["audio"] = audio
        if chunk_configs is None: chunk_configs = []
        chunk_configs.append(chunk_config)
        return (
            chunk_configs,
        )






#class ChunkerOutpaintConfig:
#    @classmethod
#    def INPUT_TYPES(cls):
#        files = ["None", *sorted(get_input_filenames())]
#        return {
#            "required": {
#                "top": ("INT", {"min": 0}),
#                "right": ("INT", {"min": 0}),
#                "bottom": ("INT", {"min": 0}),
#                "left": ("INT", {"min": 0}),
#                "feather": ("INT", {"min": 0}),
#            },
#        }

#    RETURN_TYPES = ("OUTPAINT_CONFIG",)
#    RETURN_NAMES = ("outpaint_config",)
#    OUTPUT_TOOLTIPS = (
#        "Chunk outpaint config",
#    )
#    FUNCTION = "execute"
#    CATEGORY = "Chunker"
#    DESCRIPTION = "ChunkerOutpaintConfig"

#    def execute(
#        self,
#        top,
#        right,
#        bottom,
#        left,
#        feather,
#    ):
#        chunk_outpaint_config = {
#            "top": top,
#            "right": right,
#            "bottom": bottom,
#            "left": left,
#            "feather": feather,
#        }
#        return (
#            chunk_outpaint_config,
#        )

#class ChunkerChunkConfig:
#    @classmethod
#    def INPUT_TYPES(cls):
#        files = ["None", *sorted(get_input_filenames())]
#        return {
#            "required": {
#                "include_in": (["specified_chunk_only", "every_nth_chunk", "every_chunk"], {}),
#                "chunk": ("INT", {"min": 1, "tooltip": "Which chunk these settings affect"}),
#                "frame": (["start", "end"], {"default": "start", "tooltip": "Frames within the chunk that the image will appear"}),
#                "obscure": ("BOOLEAN", {"default": False, "tooltip": "Fill grey in the image inside the masked area"}),
#                "images": (files, {"default": "None", "tooltip": "Images to be chunked"}),
#                "masks": (files, {"default": "None", "tooltip": "Masks to be chunked"}),
#            },
#            "optional": {
#                "image": (files,),
#                "image_paint": (files,),
#                "chunk_config": ("CHUNK_CONFIG",),
#                "outpaint_config": ("OUTPAINT_CONFIG",),
#            },
#        }#

#    @classmethod
#    def VALIDATE_INPUTS(cls, include_in, chunk, frame, images, masks, obscure, image, image_paint):
#        # YOLO, anything goes!
#        return True

#    RETURN_TYPES = ("CHUNK_CONFIG",)
#    RETURN_NAMES = ("chunk_config",)
#    OUTPUT_TOOLTIPS = (
#        "Chunk config",
#    )
#    FUNCTION = "execute"
#    CATEGORY = "Chunker"
#    DESCRIPTION = "ChunkerChunkConfig"

#    def execute(
#        self,
#        include_in,
#        chunk,
#        frame,
#        images,
#        masks,
#        obscure,
#        image="None",
#        image_paint="None",
#        chunk_config=None,
#        outpaint_config=None,
#    ):
#        if chunk_config is None: chunk_config = []
#        chunk_config.append({
#            "outpaint_config": outpaint_config,
#            "include_in": include_in,
#            "chunk": chunk,
#            "frame": frame,
#            "images": images,
#            "masks": masks,
#            "obscure": obscure,
#            "image": image,
#            "image_paint": image_paint,
#        })
#        return (
#            chunk_config,
#        )











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
                #"chunk_configs": ("CHUNK_CONFIGS",),
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

        #images="None",
        #masks="None",
        #image="None",
        #image_paint="None",

        images=None,
        masks=None,
        audio=None,
        fps=None,
        store=None,
        chunk_configs=None,
        unique_id=None,
    ):
        s = store if store is not None else {
            "index": 0,
            "images_last_chunk_path": None,
            "masks_last_chunk_path": None,
        }

        w = None
        h = None
        fps = None

	# get total_length and fps from last video in configs
        video_length = None
        video_fps = None
        if chunk_configs is not None:
            for config in chunk_configs:
                images_path = parse_config_paths(config["paths"])[0]
                if images_path is not None and images_path.endswith(".mp4"):
                    video_length, video_fps = get_video_info(images_path)
        if video_length is not None: total_length = video_length
        if video_fps is not None: fps = video_fps
        if fps is None: fps = 30

        if mode == "Wan21" or mode == "Wan22":
            chunk_length = force_wan_length(chunk_length)
            total_length = fix_total_length(total_length, chunk_length, chunk_overlap)
            if mode == "Wan21": fps = 16
            if mode == "Wan22": fps = 24

        this_chunk_length = get_this_chunk_length(s["index"], chunk_length, chunk_overlap, total_length)

        start = s["index"] * (chunk_length - chunk_overlap)
        end = start + chunk_length
        chunk_count = math.ceil((total_length - chunk_overlap) / (chunk_length - chunk_overlap))

        #if images == "None": images = None
        #if masks == "None": masks = None
        #if image == "None": image = None
        #if image_paint == "None": image_paint = None

        out_images = []
        out_masks = []

        # get the images overlap from store file
        if s["images_last_chunk_path"] is not None:
            images_overlap = awesome_loader(s["images_last_chunk_path"], start=-chunk_overlap)[0]
            w = images_overlap.shape[2]
            h = images_overlap.shape[1]
            out_images.append(images_overlap)
            start_frame = images_overlap[0].unsqueeze(0)

        # get the masks overlap from store file
        if s["masks_last_chunk_path"] is not None:
            imasks_overlap = awesome_loader(s["masks_last_chunk_path"], start=-chunk_overlap)[0]
            masks_overlap = image_to_mask(imasks_overlap)
            out_masks.append(masks_overlap)


        if (mode == "Wan21" or mode == "Wan22") and (count(out_images) > count(out_masks)):
            black_panel = panel_mask(w, h, 0)
            out_masks.append(torch.cat([black_panel] * (count(out_images) - count(out_masks)))) # add same amount of black masks to masks



        # use chunk_configs
        start_image = None
        end_image = None
        start_mask = None
        end_mask = None
        #obscure_start = False
        #obscure_end = False
        #obscure_all = False
        images = None
        masks = None
        if chunk_configs is not None:
            for config in chunk_configs:
                if (
                    (config["include_in"] == "specified_chunk_only" and (s["index"] + 1) == config["chunk"])
                    or
                    (config["include_in"] == "every_nth_chunk" and ((s["index"] + 1) % config["chunk"] == 0))
                    or
                    (config["include_in"] == "every_chunk")
                ):
                    images_path, masks_path, mask_maskeditor_path, paint_maskeditor_path = parse_config_paths(config["paths"])

                    #images_chunk = None
                    if images_path is not None:
                        images_chunk, images_fps, images_total_length = awesome_loader(images_path, start + count(out_images), end)
                        if total_length == 0: total_length = images_total_length
                        if images_fps is not None: fps = images_fps
                        if w is None: w = images_chunk.shape[2]
                        if h is None: h = images_chunk.shape[1]
                        if config["frame"] == "start":
                            #obscure_start = config["obscure"]
                            start_image = images_chunk[0].unsqueeze(0)
                        if config["frame"] == "end":
                            #obscure_end = config["obscure"]
                            end_image = images_chunk[0].unsqueeze(0)
                        if config["frame"] == "every":
                            #obscure_all = config["obscure"]
                            images = images_chunk
                            #out_images.append(images_chunk)

                    #masks_chunk = None
                    if masks_path is not None:
                        imasks_chunk = awesome_loader(masks_path, start + count(out_masks), end)[0]
                        masks_chunk = image_to_mask(imasks_chunk)
                        if config["frame"] == "every":
                            masks = masks_chunk
                            #out_masks.append(masks_chunk)

                    #mask_maskeditor = None
                    if mask_maskeditor_path is not None:
                        mask_maskeditor = awesome_loader(mask_maskeditor_path, return_masks=True)[0]
                        if config["frame"] == "start": start_mask = mask_maskeditor
                        if config["frame"] == "end": end_mask = mask_maskeditor

                    # apply outpainting config
                    #if start_image is not None and config["outpaint_config"] is not None:
                    #    start_image, start_mask = expand_image(
                    #        start_image,
                    #        config["outpaint_config"]["left"],
                    #        config["outpaint_config"]["top"],
                    #        config["outpaint_config"]["right"],
                    #        config["outpaint_config"]["bottom"],
                    #        config["outpaint_config"]["feather"],
                    #        start_mask,
                    #    )
                    #if end_image is not None and config["outpaint_config"] is not None:
                    #    end_image, end_mask = expand_image(
                    #        end_image,
                    #        config["outpaint_config"]["left"],
                    #        config["outpaint_config"]["top"],
                    #        config["outpaint_config"]["right"],
                    #        config["outpaint_config"]["bottom"],
                    #        config["outpaint_config"]["feather"],
                    #        end_mask,
                    #    )


        if w is None: w = 512
        if h is None: h = 512

        grey_panel = panel_image(w, h, 127, 127, 127)
        white_panel = panel_mask(w, h, 255)
        black_panel = panel_mask(w, h, 0)

        # apply obscure_image
        #if obscure_start and start_image is not None and start_mask is not None: start_image = obscure_image(start_image, grey_panel, start_mask)
        #if obscure_end and end_image is not None and end_mask is not None: end_image = obscure_image(end_image, grey_panel, end_mask)
        #if obscure_all and images is not None and masks is not None: images = obscure_image(images, grey_panel, masks)



        if images is not None: out_images.append(images)
        if masks is not None: out_masks.append(masks)




        if start_image is not None:
            w = start_image.shape[2]
            h = start_image.shape[1]
            out_images.append(start_image)
            out_masks.append(start_mask if start_mask is not None else black_panel)

        fill_count_images = this_chunk_length - (1 if end_image is not None else 0) - count(out_images)
        fill_count_masks = this_chunk_length - (1 if end_mask is not None else 0) - count(out_masks)

        log("this_chunk_length", this_chunk_length)
        log("fill_count_images", fill_count_images)
        log("fill_count_masks", fill_count_masks)

        if fill_count_images > 0: out_images.append(torch.cat([grey_panel] * fill_count_images))
        if fill_count_masks > 0: out_masks.append(torch.cat([white_panel] * fill_count_masks))

        if end_image is not None:
            out_images.append(end_image)
            out_masks.append(end_mask if end_mask is not None else black_panel)













        # get images chunk from "images" input file
        #if images is not None:
        #    images_path_full = os.path.join(folder_paths.get_input_directory(), images)
        #    images_chunk, images_fps, images_total_length = awesome_loader(images_path_full, start + count(out_images), end)
        #    if images_fps is not None: fps = images_fps
        #    if total_length == 0: total_length = images_total_length
        #    w = images_chunk.shape[2]
        #    h = images_chunk.shape[1]
        #    if images_total_length > 1 or images_total_length == 1 and s["index"]==0: out_images.append(images_chunk)
        #    if mode == "Wan" and images_total_length == 1 and s["index"]==0:
        #        out_masks.append(black_panel) # add 1 black mask to masks (for i2v)

        # get the mask from the mask editor for first chunk only
        #if image is not None and s["index"] == 0:
        #    if " [input]" in image:
        #        mask_editor_filename = image.replace("clipspace/", "").replace(" [input]", "")
        #        path_full = os.path.join(folder_paths.get_input_directory(), 'clipspace', mask_editor_filename)
        #    if " [temp]" in image:
        #        mask_editor_filename = image.replace(" [temp]", "")
        #        path_full = os.path.join(folder_paths.get_temp_directory(), mask_editor_filename)
        #    mask_maskeditor = awesome_loader(path_full, return_masks=True)[0]
        #    out_masks.append(mask_maskeditor)

        # get masks chunk from input file
        #if masks is not None:
        #    masks_path_full = os.path.join(folder_paths.get_input_directory(), masks)
        #    imasks_chunk = awesome_loader(masks_path_full, start + count(out_masks), end)[0]
        #    masks_chunk = image_to_mask(imasks_chunk)
        #    out_masks.append(masks_chunk)

        # do some stuff for Wan
        if mode == "Wan21" or mode == "Wan22":
             #grey_panel = panel_image(w, h, 128, 128, 128)
             grey_panel = torch.full((1, w, h, 3), 0.5)
             white_panel = panel_mask(w, h, 255)

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
            #"audio": images if images is not None and images.endswith(".mp4") else None,
            "fps": fps,
        }

        ui_values = {
            "output_label_values": {
                "images": count(out_images),
                "masks": count(out_masks),
                "audio": get_audio_length(None), # TODO: chop up audio from input video or overlap
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
                None, # TODO: audio
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

        if audio is not None:
            s["audio_chunks"].append(audio)

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

            out_audio_torch = None # its not a tensor, rename this variable
            if len(s["audio_chunks"]) > 0:
                out_audio_torch = concat_audios(s["audio_chunks"])
                # load_audio(d["audio"]) if d["audio"] is not None else None,

            ui_values = {
                "input_label_values": {
                    "images": len(images) if images is not None else 0,
                    "masks": len(masks) if masks is not None else 0,
                    #"audio": f"{audio["waveform"].shape[2] / audio["sample_rate"]:.4f}s" if audio is not None else 0,
                    "audio": get_audio_length(audio),
                },
                "output_label_values": {
                    "images": len(out_images_torch) if out_images_torch is not None else 0,
                    "masks": len(out_masks_torch) if out_masks_torch is not None else 0,
                    #"audio": f"{out_audio_torch["waveform"].shape[2] / out_audio_torch["sample_rate"]:.4f}s" if out_audio_torch is not None else 0,
                    "audio": get_audio_length(out_audio_torch),
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
                    out_audio_torch,
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
                #"audio": f"{audio["waveform"].shape[2] / audio["sample_rate"]:.4f}s" if audio is not None else 0,
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
