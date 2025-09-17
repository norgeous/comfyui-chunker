# https://stackoverflow.com/a/77782755

import torch
import numpy as np
import av
from PIL import Image, ImageOps, ImageSequence
# import folder_paths


def pillow(fn, arg):
    prev_value = None
    try:
        x = fn(arg)
    #except (OSError, UnidentifiedImageError, ValueError): #PIL issues #4472 and #2445, also fixes ComfyUI issue #3416
        #prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        #ImageFile.LOAD_TRUNCATED_IMAGES = True
        #x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
    return x

# https://github.com/IuvenisSapiens/ComfyUI_Qwen2_5-VL-Instruct/blob/main/util_nodes.py
def load_image_advanced(image_path):
    # image_path = folder_paths.get_annotated_filepath(image)

    img = pillow(Image.open, image_path)

    output_images = []
    output_masks = []
    w, h = None, None

    excluded_formats = ["MPO"]

    for i in ImageSequence.Iterator(img):
        i = pillow(ImageOps.exif_transpose, i)

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
        #else:
            #mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu") #.permute(1, 2, 0) #.permute(0, 3, 1, 2)

    #if len(output_images) > 1 and img.format not in excluded_formats:
        #output_image = torch.cat(output_images, dim=0)
        #output_mask = torch.cat(output_masks, dim=0)
    #else:
        #output_image = output_images[0]
        #output_mask = output_masks[0]

    if len(output_masks) > 0: return torch.cat(output_masks) # .unsqueeze(0) #.toImage() # if we decoded a mask from alpha, discard the image?

    return torch.cat(output_images) #.reshape((1,)+torch.cat(output_images).shape)

def load_video_chunk(video_path, start_n, end_n):
    container = av.open(video_path)
    stream = container.streams.video[0]
    # duration = container.duration / av.time_base
    fps = stream.average_rate
    total_length = stream.frames

    if end_n is None: end_n = total_length # missing end fix
    if start_n < 0: start_n = total_length + start_n # negative start fix
    if end_n < 0: end_n = total_length + end_n # negative end fix

    # sanity checks
    assert start_n < end_n, "Start beyond end"
    #assert end_n <= total_length, f"End {end_n} beyond total_length {total_length}"

    start_time = start_n / fps
    end_time = end_n / fps

    # seek
    #print(f"Seeking to {start_time:.2f} seconds or earlier")
    container.seek(int(start_time / stream.time_base), stream=stream)

    # decode between frames
    all_frames_n = []
    frames = []
    for frame in container.decode(stream):
        assert frame.time == float(frame.pts * stream.time_base)

        frame_n = round(frame.pts * stream.time_base * stream.average_rate)
        all_frames_n.append(frame_n)

        if frame_n >= end_n:
            #print("end", frame_n, " > ", end_n)
            break
        elif frame_n < start_n:
            continue
        else:
            img = frame.to_ndarray(format='rgb24') # shape: (H, W, 3)
            img = torch.from_numpy(img) / 255.0 # shape: (H, W, 3)
            frames.append(img.unsqueeze(0)) # shape: (1, H, W, 3)

    #print(f"itterated {len(all_frames_n)} frames and collected {len(frames)} frames")
    #print("shape of each frame", frames[0].shape)
    return (torch.cat(frames), fps, total_length)

def awesome_loader(path, start=0, end=None):
    path = path.replace(" [input]", "")
    img_ext = ["jpeg", "jpg", "png"]
    vid_ext = ["mp4"]
    file_ext = path.split(".")[-1]
    if file_ext in img_ext:
        # image_path = folder_paths.get_annotated_filepath(image)
        image = load_image_advanced(path)
        fps = 0
        total_length = 1
        return (image, fps, total_length)
    if file_ext in vid_ext:
        frames, fps, total_length = load_video_chunk(path, start_n=start, end_n=end)
        return (frames, fps, total_length)

def load_videos_exclude_overlaps(paths, overlap, select_overlaps_from):
    all = []
    for i, filename in enumerate(paths):
        start = 0
        end = None
        if overlap > 0:
            is_first_chunk = i == 0
            is_final_chunk = i == len(paths) - 1
            if select_overlaps_from == "this_chunk" and not is_final_chunk:
                end = -overlap # skip end overlap frames
            if select_overlaps_from == "previous_chunk" and not is_first_chunk:
                start = overlap # skip start overlap frames
        all.append(awesome_loader(filename, start, end)[0])
    all_torch = torch.cat(all)
    return all_torch
