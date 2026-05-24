import os
import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from av_save import av_save

BASE_DIR = os.path.dirname(__file__)


def create_source_tensors(
    bg_color: tuple[int, int, int],
    line_num: int,
    freq: float,
    pixel_y_offset: int = 0,
    stereo: bool = False,
    video_frames: int = 30,
    audio_duration: float = 2.0,
    audio_sample_rate: int = 44100,
    video_width: int = 512,
    video_height: int = 512,
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
        audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1).unsqueeze(0)
    else:
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
    audio = {"waveform": audio_tensor, "sample_rate": audio_sample_rate}

    return {"images": images, "audio": audio}


def generate_source_video(
    output_path: str,
    bg_color: tuple[int, int, int],
    line_num: int,
    freq: float,
    pixel_y_offset: int = 0,
    stereo: bool = False,
    fps: int = 15,
    video_frames: int = 30,
    audio_duration: float = 2.0,
) -> None:
    tensors = create_source_tensors(bg_color, line_num, freq, pixel_y_offset, stereo, video_frames, audio_duration)
    av_save(
        images=tensors["images"],
        audio=tensors["audio"],
        output_path=output_path,
        fps=fps,
    )


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
    audio_left = audio_right = None
    container = av.open(video_path)
    if container.streams.audio:
        audio_stream = container.streams.audio[0]
        is_stereo = audio_stream.channels == 2
        if is_stereo:
            all_samples = []
            for frame in container.decode(audio_stream):
                plane = frame.planes[0]
                arr = np.frombuffer(plane, np.int16).copy()
                all_samples.append(arr)
            all_samples = np.concatenate(all_samples)
            audio_left = all_samples[0::2]
            audio_right = all_samples[1::2]
        else:
            for frame in container.decode(audio_stream):
                samples = frame.to_ndarray()
                audio_samples.append(samples)

    container.close()

    return frames, audio_samples, audio_left, audio_right, video_stream


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


def plot_audio_waveform(waveform, sample_rate, fps, title, output_path):
    plt.style.use('dark_background')
    waveform = waveform[0]
    num_channels = waveform.shape[0]
    num_samples = waveform.shape[1]

    width = min(40, max(12, num_samples / 5000))
    fig, axes = plt.subplots(num_channels, 1, figsize=(width, 4 * num_channels), squeeze=False)
    fig.suptitle(title, fontsize=14)

    for i in range(num_channels):
        axes[i, 0].plot(np.arange(num_samples), waveform[i].numpy(), linewidth=0.3)
        axes[i, 0].set_xlabel('Samples')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].set_title(f'Channel {i + 1}')

        tick_locations = np.arange(0, num_samples + 1, sample_rate)
        axes[i, 0].set_xticks(tick_locations)
        axes[i, 0].set_xticklabels([int(loc) for loc in tick_locations])

        interval = sample_rate / fps
        for sample_pos in np.arange(0, num_samples, interval):
            axes[i, 0].axvline(x=sample_pos, color='red', linestyle='--', alpha=0.5)

    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()


def pytest_sessionfinish(session, exitstatus):
    from av_load import av_load

    output_dir = os.path.join(BASE_DIR, "test-output")
    if not os.path.exists(output_dir):
        return

    for filename in sorted(os.listdir(output_dir)):
        if not (filename.endswith(".mp4") or filename.endswith(".webm")):
            continue
        video_path = os.path.join(output_dir, filename)
        _, audio, fps = av_load(video_path)
        if audio is None:
            continue
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        name = os.path.splitext(filename)[0]
        out = os.path.join(output_dir, f"{name}_waveform.png")
        plot_audio_waveform(waveform, sample_rate, fps, f"Audio Waveform: {name}", out)


@pytest.fixture(scope="session")
def output_dir():
    path = os.path.join(BASE_DIR, "test-output")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def source_dir():
    path = os.path.join(BASE_DIR, "test-source")
    os.makedirs(path, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def source_videos(source_dir):
    sources = [
        ("source1.mp4", (255, 0, 0), 1, 220, 0, False, 15, 30, 2.0),
        ("source2.mp4", (0, 255, 0), 2, 440, 10, False, 15, 30, 2.0),
        ("source3.mp4", (0, 0, 255), 3, 880, 20, False, 15, 30, 2.0),
        ("source-long.mp4", (0, 255, 255), 1, 10, 0, False, 15, 60, 4),
        ("source1_stereo.mp4", (255, 0, 0), 1, 220, 0, True, 15, 30, 2.0),
        ("source2_stereo.mp4", (0, 255, 0), 2, 440, 10, True, 15, 30, 2.0),
        ("source3_stereo.mp4", (0, 0, 255), 3, 880, 20, True, 15, 30, 2.0),
    ]

    for filename, color, line_num, freq, pixel_y_offset, stereo, fps, video_frames, audio_duration in sources:
        path = os.path.join(source_dir, filename)
        generate_source_video(path, color, line_num, freq, pixel_y_offset, stereo, fps, video_frames, audio_duration)

    return {
        "source1": os.path.join(source_dir, "source1.mp4"),
        "source2": os.path.join(source_dir, "source2.mp4"),
        "source3": os.path.join(source_dir, "source3.mp4"),
        "source-long": os.path.join(source_dir, "source-long.mp4"),
        "source1_stereo": os.path.join(source_dir, "source1_stereo.mp4"),
        "source2_stereo": os.path.join(source_dir, "source2_stereo.mp4"),
        "source3_stereo": os.path.join(source_dir, "source3_stereo.mp4"),
    }
