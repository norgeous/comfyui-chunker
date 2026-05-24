from fractions import Fraction
from enum import Enum
from typing import Optional
import av
import numpy as np
import torch
import comfy.utils

class Profile(Enum):
    HQ = "hq"
    WEB = "web"


PROFILE_SETTINGS = {
    Profile.HQ: {
        "file_extension": ".mp4",
        "video_codec": "h264",
        "video_pixel_format": "yuv420p",
        "video_options": {"preset": "slow", "crf": "10"},
        "input_frame_format": "rgb24",
        "audio_codec": "pcm_s16le",
        "audio_options": {},
        # "masks_as": "alpha",
    },
    Profile.WEB: {
        "file_extension": ".webm",
        "video_codec": "vp9",
        "video_pixel_format": "yuva420p",
        "video_options": {"crf": "30"},
        "input_frame_format": "rgb24",
        "audio_codec": "vorbis",
        "audio_options": {"strict": "-2"},
        # "masks_as": "stream2",
    },
}


def av_save(
    images: Optional[torch.Tensor] = None,
    audio: Optional[dict] = None,
    output_path: str = "output",
    fps: float = 30.0,
    profile: Profile = Profile.HQ,
) -> str:
    if images is None and audio is None:
        raise ValueError("At least one of images or audio must be provided")

    settings = PROFILE_SETTINGS[profile]

    if output_path.endswith(settings["file_extension"]):
        output_path = output_path[:-len(settings["file_extension"])]

    with av.open(f"{output_path}{settings['file_extension']}", mode='w') as container:
        video_stream = None
        audio_stream = None
        audio_frame = None

        if images is not None:
            count, H, W = images.shape[0], images.shape[1], images.shape[2]
            fps_fraction = Fraction(f"{fps:.6f}")
            video_stream = container.add_stream(settings["video_codec"], rate=fps_fraction)
            video_stream.thread_count = 0
            video_stream.thread_type = "AUTO"
            video_stream.pix_fmt = settings["video_pixel_format"]
            video_stream.options = settings["video_options"]
            video_stream.width = W
            video_stream.height = H
            video_stream.time_base = Fraction(1, int(fps))

        if audio is not None:
            waveform = audio["waveform"]
            audio_ndarray = (waveform.squeeze(0).cpu().numpy() * np.iinfo(np.int16).max).astype(np.int16)
            is_stereo = audio_ndarray.ndim > 1 and audio_ndarray.shape[0] == 2
            audio_stream = container.add_stream(settings["audio_codec"], rate=int(audio["sample_rate"]))
            audio_stream.options = settings["audio_options"]
            audio_stream.layout = 'stereo' if is_stereo else 'mono'
            audio_stream.time_base = Fraction(1, int(audio["sample_rate"]))
            audio_stream.bit_rate = audio["sample_rate"] * 2 * (2 if is_stereo else 1)
            audio_frame = av.AudioFrame.from_ndarray(audio_ndarray.T.reshape(1, -1), format='s16', layout='stereo' if is_stereo else 'mono')
            audio_frame.rate = audio["sample_rate"]
            audio_frame.pts = 0
            audio_frame.time_base = Fraction(1, int(audio["sample_rate"]))

        if video_stream is not None:
            pbar = comfy.utils.ProgressBar(count)
            for i in range(count):
                img = images[i]
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                img = (img * 255).cpu().numpy().astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format=settings["input_frame_format"])
                frame = frame.reformat(format=settings["video_pixel_format"])
                frame.pts = i
                for packet in video_stream.encode(frame):
                    container.mux(packet)
                pbar.update(1)

            for packet in video_stream.encode():
                container.mux(packet)

        if audio_stream is not None:
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)

            for packet in audio_stream.encode():
                container.mux(packet)

    return f"{output_path}{settings['file_extension']}"
