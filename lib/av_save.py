from fractions import Fraction
from enum import Enum
from typing import Optional, Tuple
import av
import numpy as np
import torch
import comfy.utils
from .utils_comfy import get_next_save_path


class Profile(Enum):
    HQ = "hq"
    COMFY = "comfy"
    WEBRGB = "webrgb"
    WEBRGBA = "webrgba"


PROFILE_SETTINGS = {
    Profile.HQ: {
        "file_extension": "mp4",
        "video_codec": "h264",
        "video_pixel_format": "yuv420p",
        "video_options": {"preset": "slow", "crf": "10"},
        "audio_codec": "pcm_s16le",
        "audio_options": {},
        "audio_bitrate": None,
        "audio_reformat": lambda w: (
            (w.squeeze(0).clamp(-1.0, 1.0).numpy() * np.iinfo(np.int16).max).astype(np.int16),
            "s16",
        ),
    },
    Profile.COMFY: {
        "file_extension": "mp4",
        "video_codec": "h264",
        "video_pixel_format": "yuv420p",
        "video_options": {"preset": "slow", "crf": "10"},
        "audio_codec": "aac",
        "audio_options": {},
        "audio_bitrate": 192000,
        "audio_reformat": lambda w: (
            w.squeeze(0).clamp(-1.0, 1.0).numpy().astype(np.float32),
            "flt",
        ),
    },
    Profile.WEBRGB: {
        "file_extension": "mp4",
        "video_codec": "h264",
        "video_pixel_format": "yuv420p",
        "video_options": {"preset": "fast", "crf": "23"},
        "audio_codec": "aac",
        "audio_options": {},
        "audio_bitrate": 192000,
        "audio_reformat": lambda w: (
            w.squeeze(0).clamp(-1.0, 1.0).numpy().astype(np.float32),
            "flt",
        ),
    },
    Profile.WEBRGBA: {
        "file_extension": "webm",
        "video_codec": "vp9",
        "video_pixel_format": "yuva420p",
        "video_options": {"crf": "30"},
        "audio_codec": "vorbis",
        "audio_options": {"strict": "-2"},
        "audio_bitrate": 192000,
        "audio_reformat": lambda w: (
            w.squeeze(0).clamp(-1.0, 1.0).numpy().astype(np.float32),
            "flt",
        ),
    },
}


def av_save(
    images: Optional[torch.Tensor] = None,
    masks: Optional[torch.Tensor] = None,
    audio: Optional[dict] = None,
    filename_prefix: str = "output",
    fps: float = 30.0,
    profile: Profile = Profile.HQ,
) -> Tuple[str, dict]:
    if images is None and audio is None:
        raise ValueError("At least one of images or audio must be provided")

    settings = PROFILE_SETTINGS[profile]

    output_path, frontend_data = get_next_save_path(filename_prefix, settings["file_extension"])

    with av.open(output_path, mode="w") as container:
        video_stream = None
        mask_stream = None
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

        if masks is not None:
            fps_fraction = Fraction(f"{fps:.6f}")
            mask_stream = container.add_stream(settings["video_codec"], rate=fps_fraction)
            mask_stream.thread_count = 0
            mask_stream.thread_type = "AUTO"
            mask_stream.pix_fmt = "yuv420p"
            mask_stream.options = settings["video_options"]
            mask_stream.width = W
            mask_stream.height = H
            mask_stream.time_base = Fraction(1, int(fps))

        if audio is not None:
            ndarray, fmt = settings["audio_reformat"](audio["waveform"])
            layout = "stereo" if ndarray.ndim > 1 and ndarray.shape[0] == 2 else "mono"
            audio_stream = container.add_stream(settings["audio_codec"], rate=int(audio["sample_rate"]))
            audio_stream.options = settings["audio_options"]
            audio_stream.layout = layout
            audio_stream.time_base = Fraction(1, int(audio["sample_rate"]))
            if settings["audio_bitrate"]: audio_stream.bit_rate = settings["audio_bitrate"]
            audio_frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(ndarray.T.reshape(1, -1)), format=fmt, layout=layout)
            audio_frame.rate = audio["sample_rate"]
            audio_frame.pts = 0
            audio_frame.time_base = Fraction(1, int(audio["sample_rate"]))

        if video_stream is not None:
            pbar = comfy.utils.ProgressBar(0)
            for i in range(count):
                img = images[i]
                img_np = (img * 255).cpu().numpy().astype(np.uint8)

                if mask_stream is not None and profile == Profile.WEBRGBA:
                    mask_np = (
                        masks[i].squeeze() *
                        255).cpu().numpy().astype(
                        np.uint8)
                    if mask_np.ndim == 2:
                        mask_np = mask_np[..., np.newaxis]
                    img_np = np.concatenate(
                        [img_np[..., :3], mask_np], axis=-1)
                    input_format = "rgba"
                else:
                    input_format = "rgba" if img_np.shape[2] == 4 else "rgb24"

                frame = av.VideoFrame.from_ndarray(img_np, format=input_format)
                frame = frame.reformat(format=settings["video_pixel_format"])
                frame.pts = i
                for packet in video_stream.encode(frame):
                    container.mux(packet)

                if mask_stream is not None:
                    mask_np = (masks[i].squeeze() * 255).cpu().numpy().astype(np.uint8)
                    mask_frame = av.VideoFrame.from_ndarray(mask_np, format="gray")
                    mask_frame = mask_frame.reformat(format="yuv420p")
                    mask_frame.pts = i
                    for packet in mask_stream.encode(mask_frame):
                        container.mux(packet)

                pbar.update_absolute(i + 1, count)

            for packet in video_stream.encode():
                container.mux(packet)

            if mask_stream is not None:
                for packet in mask_stream.encode():
                    container.mux(packet)

        if audio_stream is not None:
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)

            for packet in audio_stream.encode():
                container.mux(packet)

    return (output_path, frontend_data)
