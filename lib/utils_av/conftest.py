import os
import pytest
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from .av_save import av_save


def create_source_tensors(
    bg_color: tuple[int, int, int],
    line_num: int,
    freq: float,
    pixel_y_offset: int = 0,
    stereo: bool = False,
    audio_sample_rate: int = 44100,
    audio_duration: float = 2.0,
    video_width: int = 512,
    video_height: int = 512,
    video_frames: int = 30,
):
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        font = ImageFont.load_default()

    pil_frames = []
    for frame_num in range(video_frames):
        img = Image.new('RGB', (video_width, video_height), bg_color)
        draw = ImageDraw.Draw(img)

        draw.rectangle([0, 0, 299, 29], fill=(0, 0, 0))

        pixel_x = frame_num * 10
        draw.rectangle([pixel_x, pixel_y_offset, pixel_x + 9, pixel_y_offset + 9], fill=(255, 255, 255))

        y_pos = 50 + (line_num - 1) * 80
        draw.text((50, y_pos), f"{frame_num + 1:02d}", fill=(255, 255, 255), font=font)

        pil_frames.append(np.array(img))

    images = torch.from_numpy(np.stack(pil_frames)).float() / 255.0

    num_samples = int(audio_sample_rate * audio_duration)
    t = np.linspace(0, audio_duration, num_samples, endpoint=False)
    sine_wave = np.sin(2 * np.pi * freq * t) * 0.5
    audio_tensor = torch.from_numpy(sine_wave).float()
    if stereo:
        audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1)
    else:
        audio_tensor = audio_tensor.unsqueeze(0)

    return {"images": images, "audio": {"waveform": audio_tensor, "sample_rate": audio_sample_rate}}


def generate_source_videos(
    output_path: str,
    bg_color: tuple[int, int, int],
    line_num: int,
    freq: float,
    pixel_y_offset: int = 0,
    stereo: bool = False,
    fps: int = 15,
) -> None:
    tensors = create_source_tensors(bg_color, line_num, freq, pixel_y_offset, stereo)
    av_save(
        images=tensors["images"],
        audio=tensors["audio"],
        output_path=output_path,
        fps=fps,
    )


@pytest.fixture(scope="session")
def output_dir():
    os.makedirs("test-output", exist_ok=True)
    return "test-output"


@pytest.fixture(scope="session")
def source_dir():
    os.makedirs("test-source", exist_ok=True)
    return "test-source"


@pytest.fixture(scope="session")
def source_videos(source_dir):
    sources = [
        ("source1.mp4", (255, 0, 0), 1, 220, 0, False, 15),
        ("source2.mp4", (0, 255, 0), 2, 440, 10, False, 15),
        ("source3.mp4", (0, 0, 255), 3, 880, 20, False, 15),
    ]

    for filename, color, line_num, freq, pixel_y_offset, stereo, fps in sources:
        path = os.path.join(source_dir, filename)
        generate_source_videos(path, color, line_num, freq, pixel_y_offset, stereo, fps)

    return {
        "source1": "test-source/source1.mp4",
        "source2": "test-source/source2.mp4",
        "source3": "test-source/source3.mp4",
    }


def get_frame_info(video_path):
    import av
    container = av.open(video_path)
    video_stream = container.streams.video[0]

    frames = []
    for frame in container.decode(video_stream):
        img = frame.to_ndarray(format='rgb24')
        frames.append(img)

    container.close()

    audio_samples = []
    container = av.open(video_path)
    if container.streams.audio:
        audio_stream = container.streams.audio[0]
        for frame in container.decode(audio_stream):
            samples = frame.to_ndarray()
            audio_samples.append(samples)

    container.close()

    return frames, audio_samples, video_stream


def analyze_audio_frequency(samples, sample_rate=44100):
    if len(samples) == 0:
        return 0

    if isinstance(samples[0], np.ndarray):
        samples = np.array([s for f in samples for s in f.flatten()])
    else:
        samples = np.array(samples).flatten()

    if len(samples) < 128:
        return 0

    window_size = min(4096, len(samples))
    window = samples[:window_size]

    fft = np.fft.rfft(window)
    freqs = np.fft.rfftfreq(window_size, 1 / sample_rate)

    peak_idx = np.argmax(np.abs(fft))
    peak_freq = freqs[peak_idx]

    return peak_freq
