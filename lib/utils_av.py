import os
import av
import torch
import numpy as np
from fractions import Fraction
from .utils_comfy import get_next_save_path
from .utils_tensor import monochrome_image, mask_to_image, image_to_mask
from comfy_extras.nodes_audio import match_audio_sample_rates
from functools import reduce
from ..enum.options import OverlapBlendModes

profiles = {
    "webm": {
        "extension": "webm",
        "video_codec": "libvpx-vp9",
        "video_pix_fmt": "yuva420p",
        "video_stream_options": {
            "crf": "15", # between 0 and 63, 20-30 is standard
            "b": "0",
            "deadline": "good",
            "cpu-used": "2",
        },
        "audio_codec": "vorbis",
        "audio_format": "s16", # previously "s16p",
    },
    "mp4": {
        "extension": "mp4",
        "video_codec": "libx264",
        "video_pix_fmt": "yuv420p",
        "video_stream_options": {
            "crf": "18", # between 0 and 51, 18 is often considered visually lossless.
            "preset": "slow",
        },
        "audio_codec": "aac",
        "audio_format": "fltp",
    },
    "mov": {
        "extension": "mov",
        "video_codec": "prores_ks",
        "video_pix_fmt": "yuva444p10le",
        "video_stream_options": {
            "qscale": "2", # between 1 and 31
        } ,
        "audio_codec": "alac",
        "audio_format": "fltp",
    },
}
profile_names = list(profiles.keys())

def f2f(frame_format):
    if frame_format.startswith(("yuva","rgba")): return "rgba"
    return "rgb24"

alpha_modes = ['rgba', '2ndStream']
def tensor_to_vstreams(images, masks, alpha_mode=alpha_modes[0]):
    vstreams = (
        [], # for potentially stream 0
        [], # for potentially stream 1
    )
    W = images.shape[2] if images is not None else (masks.shape[2] if masks is not None else None)
    H = images.shape[1] if images is not None else (masks.shape[1] if masks is not None else None)
    count = max(
        images.shape[0] if images is not None else 0,
        masks.shape[0] if masks is not None else 0,
    )

    # Create separate streams for images and masks
    if alpha_mode == "2ndStream":
        if images is not None:
            for i in range(count):
                img = images[i] if images[i] is not None else monochrome_image(W, H, 0.5)
                img = (img * 255).cpu().numpy().astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                vstreams[0].append(frame) # put rgb into stream 0
        if masks is not None:
            imasks = mask_to_image(masks)
            for i in range(count):
                imask = imasks[i] if masks[i] is not None else monochrome_image(W, H, 1.0)
                imask = (imask * 255).cpu().numpy().astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(imask, format='rgb24')
                vstreams[1].append(frame) # put rgb into stream 1 (2nd stream)

    # Combine images and masks into RGBA format in one stream
    if alpha_mode == "rgba":
        if images is not None or masks is not None:
            for i in range(count):
                img = images[i] if images[i] is not None else monochrome_image(W, H, 0.5)
                img = (img * 255).cpu().numpy().astype(np.uint8)
                if masks is None:
                    frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                else:
                    mask = masks[i] 
                    if mask is not None:
                        mask = (mask * 255).cpu().numpy().astype(np.uint8)
                        mask = mask.reshape((H, W, 1))
                        rgba = np.concatenate([img, mask], axis=2)
                        frame = av.VideoFrame.from_ndarray(rgba, format='rgba')
                vstreams[0].append(frame) # put rgba into stream 0

    if len(vstreams[0]) > 0 and len(vstreams[1]) > 0: return vstreams
    if len(vstreams[0]) > 0 and len(vstreams[1]) == 0: return (vstreams[0],)
    return []

def tensor_to_astreams(audio):
    aframes = []
    if audio is not None:
        waveform = audio["waveform"]
        layout = 'mono' if waveform.shape[1] == 1 else 'stereo'
        audio_ndarray = (waveform.squeeze(0).cpu().numpy() * np.iinfo(np.int32).max).astype(np.int32)
        frame = av.AudioFrame.from_ndarray(audio_ndarray, format="s32p", layout=layout)
        frame.sample_rate = audio["sample_rate"]
        aframes.append(frame)
    return [aframes] if len(aframes) > 0 else []

def vstreams_to_tensor(vstreams, alpha_mode="rgba"):
    if len(vstreams) == 0: return (None, None)
    out_images = []
    out_masks = []
    if alpha_mode == "2ndStream":
        for frame in vstreams[0]:
            img = frame.to_ndarray(format=f2f(frame.format.name)) # decode frame
            img = torch.from_numpy(img) / 255.0 # convert uint8 to float32
            img = img.unsqueeze(0) # shape: (1, H, W, C)
            out_images.append(img)
        if len(vstreams) >= 2:
            for frame in vstreams[1]:
                img = frame.to_ndarray(format=f2f(frame.format.name)) # decode frame
                img = torch.from_numpy(img) / 255.0 # convert uint8 to float32
                img = img.unsqueeze(0) # shape: (1, H, W, C)
                mask = image_to_mask(img)
                out_masks.append(mask)
    if alpha_mode == "rgba":
        for frame in vstreams[0]:
            img = frame.to_ndarray(format=f2f(frame.format.name)) # decode frame
            img = torch.from_numpy(img) / 255.0 # convert uint8 to float32
            img = img.unsqueeze(0) # shape: (1, H, W, C)
            image = img[:, :, :, :3] # keep first 3 channels
            out_images.append(image)
            if img.shape[3] == 4:
                mask = img[:, :, :, 3] # keep 4th channel as mask
                out_masks.append(mask)
    return (
        torch.cat(out_images) if len(out_images) > 0 else None,
        torch.cat(out_masks) if len(out_masks) > 0 else None,
    )

def astreams_to_tensor(astreams):
    if len(astreams) == 0: return None
    out_audio = []
    for frame in astreams[0]:
        audio = frame.to_ndarray() # shape: (C, S)
        scaling_factor = 1.0
        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            scaling_factor = float(max(abs(info.min), abs(info.max)))
        audio = audio.astype(np.float32) / scaling_factor # shape: (C, S)
        audio = torch.from_numpy(audio) # shape: (C, S)
        out_audio.append(audio)
    if len(out_audio) == 0: return None
    out_audio_torch = torch.cat(out_audio, dim=1).unsqueeze(0) # shape: (1, C, S) - one batch with all samples
    audio_dict = {
        "waveform": out_audio_torch,
        "sample_rate": astreams[0][0].sample_rate
    }        
    return audio_dict

def load_streams(path=None, start_n=None, end_n=None):
    vstreams = {}
    astreams = {}
    fps = None
    frames = []
    with av.open(path) as container:
        fps = container.streams.video[0].average_rate if len(container.streams.video) > 0 else 30
        # vcount = container.streams.video[0].frames if len(container.streams.video) > 0 else None # unreliable
        # vcount might be None, if no vstream or because stream.frames might report wrong value 0
        vcount = 0
        for packet in container.demux():
            for frame in packet.decode():
                frames.append(frame)
                if isinstance(frame, av.video.frame.VideoFrame) and packet.stream.index == 0:
                    vcount += 1

    with av.open(path) as container:
        if start_n is None: start_n = 0 # missing start fix
        if end_n is None: end_n = vcount # missing end fix
        if end_n == 0: end_n = vcount # zero end fix
        if start_n < 0: start_n = vcount + start_n # negative start fix
        if end_n < 0: end_n = vcount + end_n # negative end fix
        start_t = float(round(start_n / fps, 3))
        end_t = float(round(end_n / fps, 3))
        # print(vcount, start_n, start_t, end_n, end_t)
        for packet in container.demux():
            stream_index = packet.stream.index
            for frame in packet.decode():
                isv = isinstance(frame, av.video.frame.VideoFrame)
                ftype = 'V' if isv else 'A'
                # print(ftype if frame.time >= start_t and frame.time <= end_t else ftype.lower(), end='')
                # base = (1 / (1/1000)) / fps
                # print('dts',frame.dts, 'pts',frame.pts, 'time',frame.time, 'base',frame.time_base, frame.sample_rate if not isv else 'no', start_t, end_t, frame.time >= start_t, frame.time <= end_t)
                if isinstance(frame, av.video.frame.VideoFrame):
                    if frame.time >= start_t and frame.time <= end_t:
                        if stream_index not in vstreams: vstreams[stream_index] = []
                        vstreams[stream_index].append(frame)
                if isinstance(frame, av.audio.frame.AudioFrame):
                    if frame.time >= start_t and frame.time <= end_t:
                        if stream_index not in astreams: astreams[stream_index] = []
                        astreams[stream_index].append(frame)
    out_vstreams = []
    for k in vstreams: out_vstreams.append(vstreams[k])
    out_astreams = []
    for k in astreams: out_astreams.append(astreams[k])
   
    return out_vstreams, out_astreams, fps

def load(path=None, alpha_mode="rgba", start_n=None, end_n=None):
    vstreams, astreams, fps = load_streams(path=path, start_n=start_n, end_n=end_n)
    # print(vstreams)
    images, masks = vstreams_to_tensor(vstreams, alpha_mode=alpha_mode)
    audio = astreams_to_tensor(astreams)
    return (images, masks, audio, fps)

def save(images=None, masks=None, audio=None, fps=30, profile=profile_names[0], alpha_mode=alpha_modes[0], filename_prefix="video/chunker/save"):
    if images is None and masks is None and audio is None:
        raise ValueError("At least one of images, masks, or audio must be provided.")
    
    vstreams = tensor_to_vstreams(images, masks, alpha_mode=alpha_mode)
    astreams = tensor_to_astreams(audio)

    out_path, frontend_data = get_next_save_path(filename_prefix, profiles[profile]["extension"])
    with av.open(out_path, mode='w') as container:
        W = images.shape[2] if images is not None else (masks.shape[2] if masks is not None else None)
        H = images.shape[1] if images is not None else (masks.shape[1] if masks is not None else None)

        # Video stream setup
        out_vstreams = []
        for i in enumerate(vstreams):
            video_stream = container.add_stream(profiles[profile]["video_codec"], rate=Fraction(f"{fps:.6f}"))
            video_stream.pix_fmt = profiles[profile]["video_pix_fmt"]
            video_stream.options = profiles[profile]["video_stream_options"]
            video_stream.width = W
            video_stream.height = H
            out_vstreams.append(video_stream)

        # Audio stream setup
        out_astreams = []
        for i in enumerate(astreams):
            audio_stream = container.add_stream(profiles[profile]["audio_codec"], rate=int(audio["sample_rate"]))
            out_astreams.append(audio_stream)

        # NOTE: all av streams must be added to container before any packets are sent into any stream

        # push each vframes_stream into streams
        for i, vstream in enumerate(vstreams):
            for frame in vstream:
                for packet in out_vstreams[i].encode(frame):
                    container.mux(packet)
        
            # Flush the video stream encoder
            for packet in out_vstreams[i].encode():
                container.mux(packet)

        # push each audio packet to audio stream
        for i, astream in enumerate(astreams):
            for frame in astream:
                for packet in out_astreams[i].encode(frame):
                    container.mux(packet)

            # Flush the audio stream encoder
            for packet in out_astreams[i].encode():
                container.mux(packet)

    return out_path, frontend_data

def mux2():
    ...

def mux(paths, filename_prefix="video/chunker/mux", overlap=0, overlap_blend_mode="this_chunk"):
    ext = os.path.splitext(paths[0])[1][1:]
    out_path, frontend_data = get_next_save_path(filename_prefix, ext)
    with av.open(out_path, mode="w") as output:
        out_vstreams = []
        out_astreams = []
        v_dts_offsets = []
        a_dts_offsets = []
        for i, path in enumerate(paths):
            vpacketstreams = []
            apacketstreams = []
            with av.open(path) as input:
                vstreams = input.streams.video
                astreams = input.streams.audio

                # when 1st video is open, setup output streams same as first video
                if i == 0:
                    for j, vstream in enumerate(vstreams):
                        if len(out_vstreams) == j:
                            # out_vstream = output.add_stream(vstream.codec_context.name, rate=Fraction(f"{vstream.average_rate:.6f}"))
                            # out_vstream.pix_fmt = vstream.codec_context.pix_fmt
                            # out_vstream.options = vstream.options
                            # out_vstream.width = vstream.codec_context.width
                            # out_vstream.height = vstream.codec_context.height
                            out_vstream = output.add_stream_from_template(vstream)
                            out_vstreams.append(out_vstream)
                            v_dts_offsets.append(0)
                    for j, astream in enumerate(astreams):
                        if len(out_astreams) == j:
                            out_astream = output.add_stream(astream.codec_context.name, astream.codec_context.rate)
                            out_astreams.append(out_astream)
                            a_dts_offsets.append(0)

                # demux packets in all input streams
                for packet in input.demux():
                    if packet.dts is None: continue # skip the "flushing" packets
                    if packet.stream.type == 'video':
                        j = packet.stream_index
                        if len(vpacketstreams) == j: vpacketstreams.append([])
                        vpacketstreams[j].append(packet)
                    if packet.stream.type == 'audio':
                        j = packet.stream_index - len(vstreams)
                        if len(apacketstreams) == j: apacketstreams.append([])
                        apacketstreams[j].append(packet)

            is_first_chunk = i == 0
            is_final_chunk = i == len(paths) - 1
            start_n = overlap if overlap_blend_mode == "previous_chunk" and not is_first_chunk else None
            end_n = -overlap if overlap_blend_mode == "this_chunk" and not is_final_chunk else None

            vcount = len(vpacketstreams[0])
            if start_n is None: start_n = 0 # missing start fix
            if end_n is None: end_n = vcount # missing end fix
            if end_n == 0: end_n = vcount # zero end fix
            if start_n < 0: start_n = vcount + start_n # negative start fix
            if end_n < 0: end_n = vcount + end_n # negative end fix

            # mux video packets
            # fps = out_vstreams[0].average_rate
            # base = (1 / (1/1000)) / fps
            # print('fps', fps)
            # print('base', base)
            # print('ovs.time_base', out_vstreams[0].time_base)
            # start_dts = int(start_n * base)
            # end_dts = int(end_n * base)
            # print('start/end dts', start_dts, end_dts)


            fps = vstreams[0].average_rate

            max_dts = 0
            for j, packets in enumerate(vpacketstreams):
                # print()
                # print("file",i,'input vstream',j, "length", len(packets))
                for k, packet in enumerate(packets):
                    # fps = out_vstreams[0].average_rate
                    base = (1 / packet.time_base) / fps
                    start_dts = int(start_n * base)
                    end_dts = int(end_n * base)
                    if packet.dts < start_dts or packet.dts >= end_dts:
                        # print('v', end='')
                        # print(packet.dts, end='')
                        continue # skip trimmed packets
                    # print('V', end='')




                    # packet.dts += v_dts_offsets[j]
                    # packet.dts = v_dts_offsets[j] + (base * k)
                    # packet.pts = packet.dts
                    # packet.dts = None
                    # packet.pts = None
                    print(packet, packet.dts, packet.pts, packet.time_base)




                    packet.stream = out_vstreams[j]
                    # print(f"{packet.dts},", end="")
                    output.mux(packet)
                    max_dts = max(max_dts, packet.dts)
                v_dts_offsets[j] = max_dts + base
            
            # mux audio packets
            max_dts = 0
            for j, packets in enumerate(apacketstreams):
                # print()
                # print("file",i,'input astream',j, "length", len(packets))
                for packet in packets:
                    if packet.dts < start_dts or packet.dts >= end_dts: continue # skip trimmed packets
                    # packet.dts += a_dts_offsets[j]
                    # packet.pts = packet.dts
                    packet.stream = out_astreams[j]
                    # print(f"{packet.dts},", end="")
                    output.mux(packet)
                    max_dts = max(max_dts, packet.dts)
                a_dts_offsets[j] = max_dts

        # todo: do we need to flush output streams?

    return out_path, frontend_data

def concat_audio(audio1, audio2):
    waveform_1 = audio1["waveform"]
    waveform_2 = audio2["waveform"]
    sample_rate_1 = audio1["sample_rate"]
    sample_rate_2 = audio2["sample_rate"]
    if waveform_1.shape[1] == 1:
        waveform_1 = waveform_1.repeat(1, 2, 1) # Convert mono to stereo
    if waveform_2.shape[1] == 1:
        waveform_2 = waveform_2.repeat(1, 2, 1) # Convert mono to stereo
    waveform_1, waveform_2, output_sample_rate = match_audio_sample_rates(waveform_1, sample_rate_1, waveform_2, sample_rate_2)
    concatenated_audio = torch.cat((waveform_1, waveform_2), dim=2)
    return {
        "waveform": concatenated_audio,
        "sample_rate": output_sample_rate,
    }

def concat_audios(audios):
    return reduce(lambda a, b: concat_audio(a, b), audios)
