import torch
import numpy as np
import av
from PIL import Image, ImageOps, ImageSequence
import os
import folder_paths
from comfy_extras.nodes_video import CreateVideo
from comfy_extras.nodes_audio import SaveAudioMP3
from comfy_api.util import VideoContainer
from ..utils import count, image_to_mask, resize_image, resize_mask

# some from https://stackoverflow.com/a/77782755

# from https://github.com/IuvenisSapiens/ComfyUI_Qwen2_5-VL-Instruct/blob/main/util_nodes.py
def load_image_advanced(image_path):
    # image_path = folder_paths.get_annotated_filepath(image)

    img = Image.open(image_path)

    output_images = []
    output_masks = []
    w, h = None, None

    excluded_formats = ["MPO"]

    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)

        if i.mode == "I":
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")

        if len(output_images) == 0:
            w = image.size[0]
            h = image.size[1]

        if image.size[0] != w or image.size[1] != h:
            continue

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        output_images.append(image)

        if "A" in i.getbands():
            mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
            output_masks.append(mask.unsqueeze(0))


    output_image = None
    output_mask = None
    if len(output_images) > 1 and img.format not in excluded_formats:
        output_image = torch.cat(output_images)
        output_mask = torch.cat(output_masks)
    else:
        output_image = output_images[0]
        if len(output_masks) > 0 and torch.sum(output_masks[0]) > 0:
            output_mask = output_masks[0]

    #if len(output_masks) > 0 and torch.sum(output_masks[0]) > 0: return torch.cat(output_masks)

    #return torch.cat(output_images)

    return (
        output_image,
        output_mask,
    )

def get_video_info(video_path):
    sample_rate = None
    with av.open(video_path) as container:
        vstream = container.streams.video[0]
        fps = vstream.average_rate # might always be a Fraction
        total_length = vstream.frames
        if len(container.streams.audio) > 0:
            sample_rate = container.streams.audio[0].rate
    return (total_length, fps, sample_rate)

def vframes_to_tensor(frames):
    out_images = []
    for frame in frames:
        img = frame.to_ndarray(format='rgb24') # shape: (H, W, 3)
        img = torch.from_numpy(img) / 255.0 # shape: (H, W, 3)
        img = img.unsqueeze(0) # shape: (1, H, W, 3)
        out_images.append(img)
    return torch.cat(out_images)

def aframes_to_tensor(frames):
    out_audio = []
    for frame in frames:
        audio = frame.to_ndarray() # shape: (C, S)
        audio = torch.from_numpy(audio) # shape: (C, S)
        out_audio.append(audio)
    if len(out_audio) == 0: return None
    return torch.cat(out_audio, dim=1).unsqueeze(0) # shape: (1, C, S) - one batch with all samples

def vframes_to_muxable(frames):
    out_images = []
    for frame in frames:
        new_frame = av.VideoFrame.from_image(frame.to_image())
        out_images.append(new_frame)
    return out_images

def aframes_to_muxable(frames, rate):
    out_audio = []
    for frame in frames:
        new_frame = av.AudioFrame.from_ndarray(frame.to_ndarray(), format='fltp', layout='stereo')
        new_frame.sample_rate = rate # needed!
        out_audio.append(new_frame)
    return out_audio

def load_video_chunk(path, start_n, end_n):
    out_vframes = []
    out_aframes = []
    with av.open(path) as container:
        vstream = container.streams.video[0] if len(container.streams.video) > 0 else None
        astream = container.streams.audio[0] if len(container.streams.audio) > 0 else None
        total_length = vstream.frames
        if end_n is None: end_n = total_length # missing end fix
        if start_n < 0: start_n = total_length + start_n # negative start fix
        if end_n < 0: end_n = total_length + end_n # negative end fix
        vcount = 0
        for frame in container.decode(vstream, astream):
            if vcount >= start_n and vcount < end_n:
                if isinstance(frame, av.video.frame.VideoFrame): out_vframes.append(frame)
                if isinstance(frame, av.audio.frame.AudioFrame): out_aframes.append(frame)
            if isinstance(frame, av.video.frame.VideoFrame): vcount += 1
            if len(out_vframes) >= end_n - start_n: break
    return (
        out_vframes,
        out_aframes,
    )

def load_audio_chunk(path, start_n, end_n):
    out_aframes = []
    with av.open(path) as container:
        astream = container.streams.audio[0] if len(container.streams.audio) > 0 else None
        total_length = vstream.frames
        if end_n is None: end_n = total_length # missing end fix
        if start_n < 0: start_n = total_length + start_n # negative start fix
        if end_n < 0: end_n = total_length + end_n # negative end fix
        vcount = 0
        for frame in container.decode(vstream, astream):
            if vcount >= start_n and vcount < end_n:
                if isinstance(frame, av.video.frame.VideoFrame): out_vframes.append(frame)
                if isinstance(frame, av.audio.frame.AudioFrame): out_aframes.append(frame)
            if isinstance(frame, av.video.frame.VideoFrame): vcount += 1
            if len(out_vframes) >= end_n - start_n: break
    return (
        out_vframes,
        out_aframes,
    )

def media_loader(images, masks, image, image_paint):
    if images == "None": images = None
    if masks == "None": masks = None
    if image == "None": image = None
    if image_paint == "None": image_paint = None

    if images is not None: images = folder_paths.get_annotated_filepath(images)
    if masks is not None: masks = folder_paths.get_annotated_filepath(masks)
    if image is not None: image = folder_paths.get_annotated_filepath(image)
    if image_paint is not None: image_paint = folder_paths.get_annotated_filepath(image_paint)

    out_images = []
    out_masks = []
    out_audio = []
    fps = None

    w = None
    h = None

    total_length, fps, sample_rate = get_video_info(images)

    if image is not None:
        image, mask = load_image_advanced(image)
        out_images.append(image)
        out_masks.append(mask)

    #if image_paint is not None: image_paint = load_image_advanced(image_paint)[0]

    if images is not None:
        images, audio = awesome_loader(images, start=count(out_images))
        w = images.shape[2]
        h = images.shape[1]
        out_images.append(images)
        out_audio.append(audio)

    if masks is not None:
        imasks = awesome_loader(masks, start=count(out_masks))[0]
        masks = image_to_mask(imasks)
        out_masks.append(masks)

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
        out_audio_dict = {
            "waveform": torch.cat(out_audio),
            "sample_rate": sample_rate,
        }

    return (
        out_images_torch,
        out_masks_torch,
        out_audio_dict,
        fps,
    )










def awesome_loader(path, start=0, end=None, return_masks=False):
    path = path.replace(" [input]", "")
    img_ext = ["jpeg", "jpg", "png"]
    vid_ext = ["mp4"]
    aud_ext = ["mp3"]
    file_ext = path.split(".")[-1]
    if file_ext in img_ext:
        images, masks = load_image_advanced(path)
        return (images, masks)
    if file_ext in vid_ext:
        frames, audio = load_video_chunk(path, start_n=start, end_n=end)
        frames = vframes_to_tensor(frames)
        audio = aframes_to_tensor(audio)
        return (frames, audio)
    if file_ext in aud_ext:
        audio = load_audio_chunk(path, start_n=start, end_n=end)
        return (audio,)

def get_next_save_video_path(filename_prefix):
    format = "auto"
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
        filename_prefix,
        folder_paths.get_output_directory(),
    )
    file = f"{filename}_{counter:05}_.{VideoContainer.get_extension(format)}"
    full_path = os.path.join(full_output_folder, file)
    return (
        full_path,
        {
            "filename": file,
            "subfolder": subfolder,
            "type": "output",
        },
    )

def save_video(images, fps, filename_prefix, audio=None):
    format = "auto"
    codec = "auto"
    create_video_node = CreateVideo()
    video = create_video_node.execute(images, fps, audio)[0]
    full_path, frontend_data = get_next_save_video_path(filename_prefix)
    video.save_to(full_path, format=format, codec=codec, metadata=None)
    return (
        full_path,
        frontend_data,
    )

def save_audio(audio, filename_prefix):
    # format = "auto"
    # codec = "auto"
    save_audio_node = SaveAudioMP3()
    # full_path, frontend_data = get_next_save_video_path(filename_prefix)
    #print(save_audio_node)
    full_path, frontend_data = get_next_save_video_path(filename_prefix)
    full_path = full_path.replace('mp4','mp3')

    print(full_path, frontend_data)

    audio_save = save_audio_node.save_mp3(audio, filename_prefix=filename_prefix, format="mp3", quality="128k")
    # video.save_to(full_path, format=format, codec=codec, metadata=None)
    print(audio_save)

    return (
        full_path,
        frontend_data,
    )

# modified from https://stackoverflow.com/a/75429028
def quick_combine(paths, overlap, select_overlaps_from, filename_prefix):
    input1 = av.open(paths[0])
    input1_vstream = input1.streams.video[0] # if len(input1.streams.video) > 0 else None
    input1_astream = input1.streams.audio[0] if len(input1.streams.audio) > 0 else None
    input1.close()

    outpath, frontend_data = get_next_save_video_path(filename_prefix)

    with av.open(outpath, 'w') as output:
        out_vstream = output.add_stream(
            input1_vstream.codec_context.name,
            input1_vstream.codec_context.rate,
        )
        out_vstream.width = input1_vstream.codec_context.width
        out_vstream.height = input1_vstream.codec_context.height
        out_vstream.pix_fmt = input1_vstream.codec_context.pix_fmt
        out_vstream.options = {'crf': '10'}

        out_astream = None
        if input1_astream is not None:
            out_astream = output.add_stream(
                input1_astream.codec_context.name,
                input1_astream.codec_context.rate,
            )

        # load chunks
        for i, path in enumerate(paths):
            total_length = get_video_info(path)[0]
            is_first_chunk = i == 0
            is_final_chunk = i == len(paths) - 1
            start = overlap if select_overlaps_from == "previous_chunk" and not is_first_chunk else 0
            end = total_length - overlap if select_overlaps_from == "this_chunk" and not is_final_chunk else total_length
            vframes, aframes = load_video_chunk(path, start, end)
            vframes = vframes_to_muxable(vframes)
            for frame in vframes: output.mux(out_vstream.encode(frame))
            if len(aframes) > 0:
                aframes = aframes_to_muxable(aframes, input1_astream.codec_context.rate)
                for frame in aframes: output.mux(out_astream.encode(frame))

        # Flush the encoder
        out_packet = out_vstream.encode(None)
        output.mux(out_packet)
        if out_astream is not None:
            out_packet = out_astream.encode(None)
            output.mux(out_packet)

    return outpath, frontend_data
