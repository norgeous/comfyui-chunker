from fractions import Fraction
from typing import Optional

import av
import numpy as np
import torch

FILE_EXTENSION = ".mp4"
VIDEO_CODEC = "h264"
VIDEO_PIXEL_FORMAT = "yuv420p"
VIDEO_PRESET = "slow"
VIDEO_CRF = 10
INPUT_FRAME_FORMAT = "rgb24"
AUDIO_CODEC = "pcm_s16le"

def av_save(
    images: Optional[torch.Tensor] = None,
    audio: Optional[dict] = None,
    output_path: str = "output",
    fps: float = 30.0,
) -> str:
    if images is None and audio is None:
        raise ValueError("At least one of images or audio must be provided")

    if output_path.endswith(FILE_EXTENSION):
        output_path = output_path[:-len(FILE_EXTENSION)]

    with av.open(f"{output_path}{FILE_EXTENSION}", mode='w') as container:
        if audio is not None:
            waveform = audio["waveform"]
            audio_ndarray = (waveform.squeeze(0).cpu().numpy() * 32767).astype(np.int16)

            audio_stream = container.add_stream(AUDIO_CODEC, rate=int(audio["sample_rate"]))

            is_stereo = audio_ndarray.ndim > 1 and audio_ndarray.shape[0] == 2
            audio_stream.layout = 'stereo' if is_stereo else 'mono'
            audio_stream.time_base = Fraction(1, audio["sample_rate"])
            audio_stream.bit_rate = audio["sample_rate"] * 2 * (2 if is_stereo else 1)

            if is_stereo:
                audio_frame = av.AudioFrame.from_ndarray(audio_ndarray, format='s16p', layout='stereo')
            else:
                audio_frame = av.AudioFrame.from_ndarray(audio_ndarray.reshape(1, -1), format='s16', layout='mono')

            audio_frame.rate = audio["sample_rate"]
            audio_frame.pts = 0
            audio_frame.time_base = Fraction(1, audio["sample_rate"])

        if images is not None:
            H, W = images.shape[1], images.shape[2]
            count = images.shape[0]

            fps_fraction = Fraction(f"{fps:.6f}")
            video_stream = container.add_stream(VIDEO_CODEC, rate=fps_fraction)
            video_stream.pix_fmt = VIDEO_PIXEL_FORMAT
            video_stream.options = {'preset': VIDEO_PRESET, 'crf': str(VIDEO_CRF)}
            video_stream.width = W
            video_stream.height = H
            video_stream.time_base = Fraction(1, 1) / fps_fraction

            for i in range(count):
                img = images[i]
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                img = (img * 255).cpu().numpy().astype(np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format=INPUT_FRAME_FORMAT)
                frame = frame.reformat(format=VIDEO_PIXEL_FORMAT)
                frame.pts = i
                for packet in video_stream.encode(frame):
                    container.mux(packet)

            for packet in video_stream.encode():
                container.mux(packet)

        if audio is not None:
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)

            for packet in audio_stream.encode():
                container.mux(packet)

    return f"{output_path}{FILE_EXTENSION}"