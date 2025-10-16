import torch
import numpy as np
import av
from PIL import Image, ImageOps, ImageSequence
import os
import folder_paths
from comfy_extras.nodes_video import CreateVideo
from comfy_api.util import VideoContainer
from .utils import count, image_to_mask, resize_image, resize_mask

# some from https://stackoverflow.com/a/77782755

#def pillow(fn, arg):
#    prev_value = None
#    try:
#        x = fn(arg)
#    #except (OSError, UnidentifiedImageError, ValueError): #PIL issues #4472 and #2445, also fixes ComfyUI issue #3416
#        #prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
#        #ImageFile.LOAD_TRUNCATED_IMAGES = True
#        #x = fn(arg)
#    finally:
#        if prev_value is not None:
# InvertMask           ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
#    return x

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
    with av.open(video_path) as container:
        vstream = container.streams.video[0]
        fps = float(vstream.average_rate)
        total_length = vstream.frames
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

def load_video_chunk2(path, start_n, end_n):
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

#def load_video_chunk(video_path, start_n, end_n):
#    container = av.open(video_path)
#    stream = container.streams.video[0]
#    # duration = container.duration / av.time_base
#    fps = stream.average_rate
#    total_length = stream.frames

#    if end_n is None: end_n = total_length # missing end fix
#    if start_n < 0: start_n = total_length + start_n # negative start fix
#    if end_n < 0: end_n = total_length + end_n # negative end fix

    # sanity checks
#    assert start_n < end_n, "Start beyond end"
#    #assert end_n <= total_length, f"End {end_n} beyond total_length {total_length}"

#    start_time = start_n / fps
#    end_time = end_n / fps

    # seek
    #print(f"Seeking to {start_time:.2f} seconds or earlier")
#    container.seek(int(start_time / stream.time_base), stream=stream)

    # decode between frames
#    all_frames_n = []
#    frames = []
#    for frame in container.decode(stream):
#        assert frame.time == float(frame.pts * stream.time_base)

#        frame_n = round(frame.pts * stream.time_base * stream.average_rate)
#        all_frames_n.append(frame_n)

#        if frame_n >= end_n:
#            #print("end", frame_n, " > ", end_n)
#            break
#        elif frame_n < start_n:
#            continue
#        else:
#            img = frame.to_ndarray(format='rgb24') # shape: (H, W, 3)
#            img = torch.from_numpy(img) / 255.0 # shape: (H, W, 3)
#            frames.append(img.unsqueeze(0)) # shape: (1, H, W, 3)

    #print(f"itterated {len(all_frames_n)} frames and collected {len(frames)} frames")
    #print("shape of each frame", frames[0].shape)
#    out_images = torch.cat(frames)

#    assert len(out_images) == end_n - start_n, f"Length of images is {len(out_images)}, but wanted {end_n - start_n}"

#    return (out_images, fps, total_length)



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
    file_ext = path.split(".")[-1]
    if file_ext in img_ext:
        # image_path = folder_paths.get_annotated_filepath(image)
        images, masks = load_image_advanced(path)
        return (images, masks)
        #fps = None
        #total_length = 1
        #if return_masks: return (masks, fps, total_length)
        #else: return (images, fps, total_length)
    if file_ext in vid_ext:
        #frames, fps, total_length = load_video_chunk(path, start_n=start, end_n=end)
        frames, audio = load_video_chunk2(path, start_n=start, end_n=end)
        frames = vframes_to_tensor(frames)
        audio = aframes_to_tensor(audio)
        if audio is not None: print("final shape", audio.shape)
        #fps = 30
        #total_length = 1000
        return (frames, audio) # , fps, total_length)

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
    video, = create_video_node.execute(images, fps, audio)
    full_path, frontend_data = get_next_save_video_path(filename_prefix)
    video.save_to(full_path, format=format, codec=codec, metadata=None)
    return (
        full_path,
        frontend_data,
    )

# modified from https://stackoverflow.com/a/75429028
#def quick_combine_old(paths, overlap, select_overlaps_from, filename_prefix):
#    input1 = av.open(paths[0])
#    input1_stream = input1.streams.video[0]
#    input1_astream = input1.streams.audio[0] if len(input1.streams.audio) > 0 else None
#    input1.close()

#    fps = input1_stream.codec_context.rate

#    outpath, frontend_data = get_next_save_video_path(filename_prefix)

#    with av.open(outpath, 'w') as output:
#        out_stream = output.add_stream(
#            input1_stream.codec_context.name,
#            input1_stream.codec_context.rate,
#        )
#        out_stream.width = input1_stream.codec_context.width
#        out_stream.height = input1_stream.codec_context.height
#        out_stream.pix_fmt = input1_stream.codec_context.pix_fmt
#        out_stream.options = {'crf': '10'}

        #print("quick_combine audio", input1_astream.codec_context.name, input1_astream.codec_context.rate)
#        out_astream = None
#        if input1_astream is not None:
#            out_astream = output.add_stream(
#                input1_astream.codec_context.name,
#                input1_astream.codec_context.rate,
#            )

#        for i, path in enumerate(paths):
#            with av.open(path) as container:
#                duration = container.streams.video[0].duration
#                time_base = container.streams.video[0].time_base
#                total_length = float((duration - time_base) * time_base)
#                start = 0
#                end = total_length
#                if overlap > 0:
#                    is_first_chunk = i == 0
#                    is_final_chunk = i == len(paths) - 1
#                    if select_overlaps_from == "this_chunk" and not is_final_chunk:
#                        end = float(total_length - (overlap * (1 / fps))) # skip end overlap frames
#                    if select_overlaps_from == "previous_chunk" and not is_first_chunk:
#                        start = float(overlap * (1 / fps)) # skip start overlap frames
#                #v_count = 0
#                #a_count = 0
#                for frame in container.decode(container.streams.video[0], container.streams.audio[0] if len(container.streams.audio) > 0 else None):
#                    if frame.time >= start and frame.time < end:
#                        if isinstance(frame, av.video.frame.VideoFrame):
#                            #v_count = v_count + 1 # temp debug
#                            #print("V >>", v_count, start, frame.time, end)
#                            output.mux(out_stream.encode(av.VideoFrame.from_image(frame.to_image()))) # decode then encode erases the pts
#                        if isinstance(frame, av.audio.frame.AudioFrame):
#                            #a_count = a_count + 1 # temp debug
#                            #print("A >>", a_count, start, frame.time, end)
#                            new_frame = av.AudioFrame.from_ndarray(frame.to_ndarray(), format='fltp', layout='stereo')
#                            new_frame.sample_rate = input1_astream.codec_context.rate
#                            output.mux(out_astream.encode(new_frame))
#                    #else: print("X >>", type(frame), start, frame.time, end)

#        # Flush the encoder
#        out_packet = out_stream.encode(None)
#        output.mux(out_packet)
#        if out_astream is not None:
#            out_packet = out_astream.encode(None)
#            output.mux(out_packet)

#    return outpath, frontend_data

def quick_combine(paths, overlap, select_overlaps_from, filename_prefix):
    input1 = av.open(paths[0])
    input1_vstream = input1.streams.video[0]
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
            vframes, aframes = load_video_chunk2(path, start, end)
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
