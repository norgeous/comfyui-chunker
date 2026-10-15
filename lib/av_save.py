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


class AVWriter:
    """Incremental av writer that encodes frames one at a time so callers can
    stream a video to disk without holding the full frame tensor in memory."""

    def __init__(
        self,
        filename_prefix: str,
        fps: float,
        profile: Profile = Profile.HQ,
        total_frames: Optional[int] = None,
        audio: Optional[dict] = None,
    ):
        self.settings = PROFILE_SETTINGS[profile]
        self.profile = profile
        self.fps = fps
        self.total_frames = total_frames
        self.output_path, self.frontend_data = get_next_save_path(
            filename_prefix, self.settings["file_extension"])
        self.container = av.open(self.output_path, mode="w")
        self.video_stream = None
        self.mask_stream = None
        self.audio_stream = None
        self.frame_index = 0
        self.pbar = comfy.utils.ProgressBar(0)

        # The audio stream must exist before any packet is muxed, but its
        # samples may only be known after all frames are written (combine),
        # so set the stream up from the first audio dict and encode later.
        if audio is not None:
            sample_rate = int(audio["sample_rate"])
            self.audio_stream = self.container.add_stream(
                self.settings["audio_codec"], rate=sample_rate)
            self.audio_stream.options = self.settings["audio_options"]
            waveform_shape = audio["waveform"].squeeze(0).shape
            self.audio_stream.layout = "stereo" if len(waveform_shape) > 1 and waveform_shape[0] == 2 else "mono"
            self.audio_stream.time_base = Fraction(1, sample_rate)
            if self.settings["audio_bitrate"]:
                self.audio_stream.bit_rate = self.settings["audio_bitrate"]

    def _add_video_streams(self, width: int, height: int, with_mask: bool):
        fps_fraction = Fraction(f"{self.fps:.6f}")
        self.video_stream = self.container.add_stream(
            self.settings["video_codec"], rate=fps_fraction)
        self.video_stream.thread_count = 0
        self.video_stream.thread_type = "AUTO"
        self.video_stream.pix_fmt = self.settings["video_pixel_format"]
        self.video_stream.options = self.settings["video_options"]
        self.video_stream.width = width
        self.video_stream.height = height
        self.video_stream.time_base = Fraction(1, int(self.fps))

        if with_mask:
            self.mask_stream = self.container.add_stream(
                self.settings["video_codec"], rate=fps_fraction)
            self.mask_stream.thread_count = 0
            self.mask_stream.thread_type = "AUTO"
            self.mask_stream.pix_fmt = "yuv420p"
            self.mask_stream.options = self.settings["video_options"]
            self.mask_stream.width = width
            self.mask_stream.height = height
            self.mask_stream.time_base = Fraction(1, int(self.fps))

    def add_frame(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.video_stream is None:
            self._add_video_streams(image.shape[1], image.shape[0], mask is not None)

        img_np = (image * 255).cpu().numpy().astype(np.uint8)

        if mask is not None and self.profile == Profile.WEBRGBA:
            mask_np = (mask.squeeze() * 255).cpu().numpy().astype(np.uint8)
            if mask_np.ndim == 2:
                mask_np = mask_np[..., np.newaxis]
            img_np = np.concatenate([img_np[..., :3], mask_np], axis=-1)
            input_format = "rgba"
        else:
            input_format = "rgba" if img_np.shape[2] == 4 else "rgb24"

        frame = av.VideoFrame.from_ndarray(img_np, format=input_format)
        frame = frame.reformat(format=self.settings["video_pixel_format"])
        frame.pts = self.frame_index
        for packet in self.video_stream.encode(frame):
            self.container.mux(packet)

        if self.mask_stream is not None:
            mask_np = (mask.squeeze() * 255).cpu().numpy().astype(np.uint8)
            mask_frame = av.VideoFrame.from_ndarray(mask_np, format="gray")
            mask_frame = mask_frame.reformat(format="yuv420p")
            mask_frame.pts = self.frame_index
            for packet in self.mask_stream.encode(mask_frame):
                self.container.mux(packet)

        self.frame_index += 1
        self.pbar.update_absolute(self.frame_index, self.total_frames)

    def add_audio(self, audio: dict):
        ndarray, fmt = self.settings["audio_reformat"](audio["waveform"])
        audio_frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(ndarray.T.reshape(1, -1)), format=fmt, layout=self.audio_stream.layout)
        audio_frame.rate = audio["sample_rate"]
        audio_frame.pts = 0
        audio_frame.time_base = Fraction(1, int(audio["sample_rate"]))

        for packet in self.audio_stream.encode(audio_frame):
            self.container.mux(packet)
        for packet in self.audio_stream.encode():
            self.container.mux(packet)

    def close(self) -> Tuple[str, dict]:
        if self.video_stream is not None:
            for packet in self.video_stream.encode():
                self.container.mux(packet)
        if self.mask_stream is not None:
            for packet in self.mask_stream.encode():
                self.container.mux(packet)
        self.container.close()
        return (self.output_path, self.frontend_data)


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

    writer = AVWriter(
        filename_prefix,
        fps,
        profile,
        total_frames=images.shape[0] if images is not None else None,
        audio=audio,
    )
    if images is not None:
        for i in range(images.shape[0]):
            writer.add_frame(images[i], masks[i] if masks is not None else None)
    if audio is not None:
        writer.add_audio(audio)
    return writer.close()
