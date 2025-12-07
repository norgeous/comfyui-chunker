import av
import torch
import numpy as np
from fractions import Fraction
from .utils_comfy import get_next_save_path

profiles = {
    "lossless": {
        "extension": "mov",
        "video_codec": "prores_ks",
        "video_pix_fmt": "yuva444p10le",
        "audio_codec": "alac",
        "audio_format": "fltp",
    },
    "web-rgba": {
        "extension": "webm",
        "video_codec": "vp9",
        "video_pix_fmt": "yuva420p",
        "audio_codec": "vorbis",
        "audio_format": "s16p",
    },
}

def vframes_to_tensor(frames):
    out_images = []
    for frame in frames:
        # img = frame.to_ndarray(format='rgb24') # shape: (H, W, 3)
        # img = torch.from_numpy(img) / 255.0 # shape: (H, W, 3)
        # img = img.unsqueeze(0) # shape: (1, H, W, 3)

        img = frame.to_ndarray(format='rgba') # shape: (H, W, 4)
        img = torch.from_numpy(img) / 255.0 # shape: (H, W, 4)
        img = img.unsqueeze(0) # shape: (1, H, W, 4)

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

def vframes_to_muxable(frames, target_pix_fmt):
    reformatter = av.video.reformatter.VideoReformatter()
    out_images = []
    for frame in frames:
        new_frame = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgba"), format="rgba") # convert to yuva420p is not supported currently
        new_frame = reformatter.reformat(new_frame, format=target_pix_fmt)
        out_images.append(new_frame)
    return out_images

def aframes_to_muxable(frames, target_format):
    resampler = av.AudioResampler(
        format=target_format,
        layout=frames[0].layout,
        rate=frames[0].sample_rate,
    )
    out_audio = []
    for frame in frames:
        new_frame = resampler.resample(frame)[0]
        new_frame.pts = None
        out_audio.append(new_frame)
    return out_audio

def load_frames(path=None, start_n=None, end_n=None):
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
    return out_vframes, out_aframes

def load(path=None, start_n=None, end_n=None):
    vframes, aframes = load_frames(path=path, start_n=start_n, end_n=end_n)
    video = vframes_to_tensor(vframes)
    audio = aframes_to_tensor(aframes)
    images = video[:, :, :, :3]
    masks = video[:, :, :, 3:4]
    if audio is not None:
        sample_rate = aframes[0].sample_rate # if len(aframes) > 0 else 44100
        audio_dict = {
            "waveform": audio,
            "sample_rate": sample_rate,
        }
    return (images, masks, audio_dict)

def save(images=None, masks=None, audio=None, fps=30, profile="lossless", filename_prefix="chunker_save"):
    if images is None and masks is None and audio is None:
        raise ValueError("At least one of images, masks, or audio must be provided.")

    out_path, frontend_data = get_next_save_path(filename_prefix, profiles[profile]["extension"])
    with av.open(out_path, mode='w') as container:
        # Video stream setup
        if images is not None or masks is not None:
            count = max(
                images.shape[0] if images is not None else 0,
                masks.shape[0] if masks is not None else 0,
            )
            W = images.shape[2] if images is not None else (masks.shape[2] if masks is not None else None)
            H = images.shape[1] if images is not None else (masks.shape[1] if masks is not None else None)

            video_stream = container.add_stream(profiles[profile]["video_codec"], rate=Fraction(fps))
            video_stream.pix_fmt = profiles[profile]["video_pix_fmt"]
            video_stream.width = W
            video_stream.height = H

        # Audio stream setup
        if audio is not None:
            sample_rate = audio["sample_rate"]
            audio_stream = container.add_stream('alac', rate=int(sample_rate))

        # Combine images and masks into RGBA format and write to video stream
        if images is not None or masks is not None:
            for i in range(count):
                img = images[i] if images is not None and images[i] is not None else torch.full((H, W, 3), 0.5)
                mask = masks[i] if masks is not None and masks[i] is not None else torch.full((H, W, 1), 1)
                img = (img * 255).cpu().numpy().astype(np.uint8)
                mask = (mask * 255).cpu().numpy().astype(np.uint8)
                if mask.ndim == 2: mask = mask.reshape((H, W, 1))
                rgba = np.concatenate([img, mask], axis=2)
                frame = av.VideoFrame.from_ndarray(rgba, format='rgba')
                for packet in video_stream.encode(frame):
                    container.mux(packet)
            
            # Flush the video stream encoders
            for packet in video_stream.encode():
                container.mux(packet)

        # Write Audio to audio stream
        if audio is not None:
            waveform = audio["waveform"]
            layout = 'mono' if waveform.shape[1] == 1 else 'stereo'
            audio_ndarray = waveform.squeeze(0).cpu().numpy().astype(np.int32)
            frame = av.AudioFrame.from_ndarray(audio_ndarray, format="s32p", layout=layout)
            frame.sample_rate = sample_rate
            for packet in audio_stream.encode(frame):
                container.mux(packet)

            # Flush the audio stream encoders
            for packet in audio_stream.encode():
                container.mux(packet)

    return out_path, frontend_data

def mux(paths, profile="lossless", filename_prefix="chunker_mux", overlap=0, select_overlaps_from="this_chunk"):
    # probe first file
    input1 = av.open(paths[0])
    input1_vstream = input1.streams.video[0] if len(input1.streams.video) > 0 else None
    input1_astream = input1.streams.audio[0] if len(input1.streams.audio) > 0 else None
    fps = input1_vstream.average_rate if input1_vstream else None
    sample_rate = input1_astream.codec_context.rate if input1_astream else None
    w = input1_vstream.codec_context.width if input1_vstream else None
    h = input1_vstream.codec_context.height if input1_vstream else None
    input1.close()

    out_path, frontend_data = get_next_save_path(filename_prefix, profiles[profile]["extension"])
    with av.open(out_path, 'w') as output:
        out_vstream = output.add_stream(profiles[profile]["video_codec"], fps)
        out_vstream.pix_fmt = profiles[profile]["video_pix_fmt"]
        out_vstream.width = w
        out_vstream.height = h

        out_astream = output.add_stream(profiles[profile]["audio_codec"], sample_rate)

        # for path in paths:
        for i, path in enumerate(paths):
            is_first_chunk = i == 0
            is_final_chunk = i == len(paths) - 1
            start_n = overlap if select_overlaps_from == "previous_chunk" and not is_first_chunk else None
            end_n = -overlap if select_overlaps_from == "this_chunk" and not is_final_chunk else None
            vframes, aframes = load_frames(path=path, start_n=start_n, end_n=end_n)
            if len(vframes) > 0:
                video = vframes_to_muxable(vframes, target_pix_fmt=profiles[profile]["video_pix_fmt"])
                for frame in video: output.mux(out_vstream.encode(frame))
            if len(aframes) > 0:
                audio = aframes_to_muxable(aframes, target_format=profiles[profile]["audio_format"])
                for frame in audio: output.mux(out_astream.encode(frame))

        # Flush the encoders
        if out_vstream is not None:
            out_packet = out_vstream.encode(None)
            output.mux(out_packet)
        if out_astream is not None:
            out_packet = out_astream.encode(None)
            output.mux(out_packet)

    return out_path, frontend_data
