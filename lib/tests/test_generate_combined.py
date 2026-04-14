import os
import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import av
from fractions import Fraction
from typing import Tuple
from generate_combined import generate_combined, BlendMode

AUDIO_SAMPLE_RATE: int = 44100
AUDIO_DURATION: float = 2.0
VIDEO_FPS: int = 15
VIDEO_WIDTH: int = 512
VIDEO_HEIGHT: int = 512
VIDEO_FRAMES: int = 30


def generate_source_videos(
    output_path: str,
    bg_color: Tuple[int, int, int],
    line_num: int,
    freq: float,
    pixel_y_offset: int = 0,
) -> None:
    output = av.open(output_path, mode='w')
    
    video_stream = output.add_stream('h264', rate=VIDEO_FPS)
    video_stream.width = VIDEO_WIDTH
    video_stream.height = VIDEO_HEIGHT
    video_stream.pix_fmt = 'yuv420p'
    video_stream.options = {'preset': 'slow', 'crf': '10', 'gop_size': '1', 'keyint_min': '1'}
    
    audio_stream = output.add_stream('pcm_s16le', rate=AUDIO_SAMPLE_RATE)
    audio_stream.layout = 'mono'
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        font = ImageFont.load_default()
    
    for frame_num in range(VIDEO_FRAMES):
        img = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), bg_color)
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([0, 0, 299, 29], fill=(0, 0, 0))
        
        pixel_x = frame_num * 10
        draw.rectangle([pixel_x, pixel_y_offset, pixel_x + 9, pixel_y_offset + 9], fill=(255, 255, 255))
        
        y_pos = 50 + (line_num - 1) * 80
        draw.text((50, y_pos), f"{frame_num + 1:02d}", fill=(255, 255, 255), font=font)
        
        frame = av.VideoFrame.from_ndarray(np.array(img), format='rgb24')
        frame = frame.reformat(format='yuv420p')
        frame.pts = frame_num
        frame.time_base = Fraction(1, VIDEO_FPS)
        
        for packet in video_stream.encode(frame):
            output.mux(packet)
    
    for packet in video_stream.encode():
        output.mux(packet)
    
    num_samples = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION)
    t = np.linspace(0, AUDIO_DURATION, num_samples, endpoint=False)
    sine_wave = np.sin(2 * np.pi * freq * t) * 0.5
    
    audio_data = (sine_wave * 32767).astype(np.int16)
    audio_frame = av.AudioFrame.from_ndarray(audio_data.reshape(1, -1), format='s16', layout='mono')
    audio_frame.rate = AUDIO_SAMPLE_RATE
    audio_frame.pts = 0
    audio_frame.time_base = Fraction(1, AUDIO_SAMPLE_RATE)
    
    for packet in audio_stream.encode(audio_frame):
        output.mux(packet)
    
    for packet in audio_stream.encode():
        output.mux(packet)
    
    output.close()


@pytest.fixture(scope="session")
def source_videos():
    """Generate source videos once per test session"""
    os.makedirs("test-source", exist_ok=True)
    
    sources = [
        ("source1.mp4", (255, 0, 0), 1, 220, 0),
        ("source2.mp4", (0, 255, 0), 2, 440, 10),
        ("source3.mp4", (0, 0, 255), 3, 880, 20),
    ]
    
    for filename, color, line_num, freq, pixel_y_offset in sources:
        path = os.path.join("test-source", filename)
        generate_source_videos(path, color, line_num, freq, pixel_y_offset)
    
    return {
        "source1": "test-source/source1.mp4",
        "source2": "test-source/source2.mp4",
        "source3": "test-source/source3.mp4",
    }


@pytest.fixture
def output_dir():
    """Create output directory for test files"""
    os.makedirs("test-output", exist_ok=True)
    return "test-output"


def get_frame_info(video_path):
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    audio_stream = container.streams.audio[0] if container.streams.audio else None
    
    frames = []
    for frame in container.decode(video_stream):
        img = frame.to_ndarray(format='rgb24')
        frames.append(img)
    
    container.close()
    
    audio_samples = []
    if audio_stream:
        container = av.open(video_path)
        audio_stream = container.streams.audio[0]
        
        for frame in container.decode(audio_stream):
            samples = frame.to_ndarray()
            audio_samples.append(samples)
        
        container.close()
    
    return frames, audio_samples, video_stream, audio_stream


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


def read_pixel_position(frame):
    """
    Read the 10x10 walking pixel position from an output frame.
    Returns (source_idx, frame_idx) based on pixel position.
    
    The pixel is a 10x10 colored square that moves 10px per frame.
    Source is determined by y-position of the pixel (0-9, 10-19, 20-29).
    We detect it by finding the brightest pixel in the expected y-range.
    
    Returns None if pixel not found.
    """
    img = frame if isinstance(frame, np.ndarray) else frame.to_ndarray(format='rgb24')
    
    bg_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    y_ranges = [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (10, 11, 12, 13, 14, 15, 16, 17, 18, 19), (20, 21, 22, 23, 24, 25, 26, 27, 28, 29)]
    bg_threshold = 20
    
    for source_idx, y_range in enumerate(y_ranges):
        expected_bg = bg_colors[source_idx]
        row = img[y_range[1]]
        
        best_bg_x = 0
        best_bg_diff = 1000
        for x in range(150, 400):
            local_bg = row[x].astype(np.int32)[:3]
            bg_diff = abs(int(local_bg[0]) - expected_bg[0]) + abs(int(local_bg[1]) - expected_bg[1]) + abs(int(local_bg[2]) - expected_bg[2])
            if bg_diff < best_bg_diff:
                best_bg_diff = bg_diff
                best_bg_x = x
        
        if best_bg_diff > bg_threshold:
            continue
        
        local_bg = row[best_bg_x].astype(np.int32)[:3]
        bg_brightness = int(local_bg[0]) + int(local_bg[1]) + int(local_bg[2])
        
        best_x = 0
        best_brightness = 0
        
        for x in range(len(img[0])):
            brightness_sum = 0
            count = 0
            for y in y_range:
                for dx in range(10):
                    if x + dx < len(img[0]):
                        pixel = img[y][x + dx].astype(np.int32)[:3]
                        brightness_sum += int(pixel[0]) + int(pixel[1]) + int(pixel[2])
                        count += 1
            
            avg_brightness = brightness_sum / count if count > 0 else 0
            
            if avg_brightness > best_brightness:
                best_brightness = avg_brightness
                best_x = x
        
        if best_brightness > bg_brightness + 25:
            frame_idx = best_x // 10
            return (source_idx, frame_idx)
    
    return None


def test_no_overlap(source_videos, output_dir):
    """Test: No overlap (3 videos, 90 frames with 0 overlap)"""
    videos = [
        str(source_videos["source1"]),
        str(source_videos["source2"]),
        str(source_videos["source3"]),
    ]
    output_path = os.path.join(output_dir, "combined-3x30i-0o-none.mp4")
    
    generate_combined(videos, output_path, 0, BlendMode.LINEAR, BlendMode.EQUAL_POWER)
    
    frames, audio_samples, _, _ = get_frame_info(output_path)
    
    assert len(frames) == 90, f"Expected 90 frames, got {len(frames)}"
    
    frames_array = np.array(frames)
    
    avg_bg_source1 = np.mean(frames_array[0:30, :, 300:], axis=(0, 1, 2))[:3]
    assert abs(avg_bg_source1[0] - 255) < 10 and avg_bg_source1[1] < 10 and avg_bg_source1[2] < 10, f"Frames 0-29 should be red"
    
    avg_bg_source2 = np.mean(frames_array[30:60, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_source2[0] < 10 and abs(avg_bg_source2[1] - 255) < 10 and avg_bg_source2[2] < 10, f"Frames 30-59 should be green"
    
    avg_bg_source3 = np.mean(frames_array[60:90, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_source3[0] < 10 and avg_bg_source3[1] < 10 and abs(avg_bg_source3[2] - 255) < 10, f"Frames 60-89 should be blue"
    
    for i in range(90):
        result = read_pixel_position(frames[i])
        
        if i < 30:
            expected_source, expected_frame = 0, i
        elif i < 60:
            expected_source, expected_frame = 1, i - 30
        else:
            expected_source, expected_frame = 2, i - 60
        
        actual_source = result[0] if result else None
        actual_frame = result[1] if result else None
        
        if result is None:
            continue
        
        assert actual_source == expected_source and actual_frame == expected_frame, \
            f"Frame {i}: expected source={expected_source}, frame={expected_frame}, got source={actual_source}, frame={actual_frame}"

    audio_220 = analyze_audio_frequency(audio_samples[0:87])
    assert 200 < audio_220 < 240, f"Audio frames 0-86 should be ~220Hz, got {audio_220}"
    
    audio_440 = analyze_audio_frequency(audio_samples[87:174])
    assert 420 < audio_440 < 460, f"Audio frames 87-173 should be ~440Hz, got {audio_440}"
    
    audio_880 = analyze_audio_frequency(audio_samples[174:])
    assert 860 < audio_880 < 900, f"Audio frames 174+ should be ~880Hz, got {audio_880}"


def test_older_only(source_videos, output_dir):
    """Test: older_only blend mode"""
    videos = [
        str(source_videos["source1"]),
        str(source_videos["source2"]),
        str(source_videos["source3"]),
    ]
    output_path = os.path.join(output_dir, "combined-3x30i-4o-older_only.mp4")
    
    generate_combined(videos, output_path, 4, BlendMode.OLDER_ONLY, BlendMode.OLDER_ONLY)
    
    frames, audio_samples, _, _ = get_frame_info(output_path)
    
    assert len(frames) == 82, f"Expected 82 frames, got {len(frames)}"
    
    frames_array = np.array(frames)
    
    avg_bg_1_30 = np.mean(frames_array[0:30, :, 300:], axis=(0, 1, 2))[:3]
    assert abs(avg_bg_1_30[0] - 255) < 10 and avg_bg_1_30[1] < 10 and avg_bg_1_30[2] < 10, f"Frames 1-30 should be red"
    
    avg_bg_31_56 = np.mean(frames_array[30:56, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_31_56[0] < 10 and abs(avg_bg_31_56[1] - 255) < 10 and avg_bg_31_56[2] < 10, f"Frames 31-56 should be green"
    
    avg_bg_57_82 = np.mean(frames_array[56:82, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_57_82[0] < 10 and avg_bg_57_82[1] < 10 and abs(avg_bg_57_82[2] - 255) < 10, f"Frames 57-82 should be blue"
    
    audio_220 = analyze_audio_frequency(audio_samples[0:87])
    assert 200 < audio_220 < 240, f"Audio frames 0-86 should be ~220Hz, got {audio_220}"
    
    audio_440 = analyze_audio_frequency(audio_samples[87:174])
    assert 420 < audio_440 < 460, f"Audio frames 87-173 should be ~440Hz, got {audio_440}"
    
    audio_880 = analyze_audio_frequency(audio_samples[174:])
    assert 860 < audio_880 < 900, f"Audio frames 174+ should be ~880Hz, got {audio_880}"
    
    for i in range(len(frames)):
        result = read_pixel_position(frames[i])
        
        if i < 30:
            expected_source, expected_frame = 0, i
        elif i < 56:
            expected_source, expected_frame = 1, i - 30 + 4
        else:
            expected_source, expected_frame = 2, i - 56 + 4
        
        if i in [26, 27, 28, 29]:
            continue
        
        actual_source = result[0] if result else None
        actual_frame = result[1] if result else None
        
        if result is None:
            continue
        
        assert actual_source == expected_source and actual_frame == expected_frame, \
            f"Frame {i}: expected source={expected_source}, frame={expected_frame}, got source={actual_source}, frame={actual_frame}"


def test_linear_blend(source_videos, output_dir):
    """Test: linear blend mode"""
    videos = [
        str(source_videos["source1"]),
        str(source_videos["source2"]),
        str(source_videos["source3"]),
    ]
    output_path = os.path.join(output_dir, "combined-3x30i-4o-linear.mp4")
    
    generate_combined(videos, output_path, 4, BlendMode.LINEAR, BlendMode.LINEAR)
    
    frames, audio_samples, _, _ = get_frame_info(output_path)
    
    assert len(frames) == 82, f"Expected 82 frames, got {len(frames)}"
    
    frames_array = np.array(frames)
    
    avg_bg_1_26 = np.mean(frames_array[0:26, :, 300:], axis=(0, 1, 2))[:3]
    assert abs(avg_bg_1_26[0] - 255) < 10 and avg_bg_1_26[1] < 10 and avg_bg_1_26[2] < 10, f"Frames 1-26 should be red"
    
    frame_27 = frames[26]
    avg_27 = np.mean(frame_27[:, 300:], axis=(0, 1))[:3]
    expected_r = 255 * 0.8
    expected_g = 255 * 0.2
    assert abs(avg_27[0] - expected_r) < 20 and abs(avg_27[1] - expected_g) < 20, f"Frame 27 should be 80% red, 20% green"
    
    frame_28 = frames[27]
    avg_28 = np.mean(frame_28[:, 300:], axis=(0, 1))[:3]
    expected_r = 255 * 0.6
    expected_g = 255 * 0.4
    assert abs(avg_28[0] - expected_r) < 20 and abs(avg_28[1] - expected_g) < 20, f"Frame 28 should be 60% red, 40% green"
    
    frame_29 = frames[28]
    avg_29 = np.mean(frame_29[:, 300:], axis=(0, 1))[:3]
    expected_r = 255 * 0.4
    expected_g = 255 * 0.6
    assert abs(avg_29[0] - expected_r) < 20 and abs(avg_29[1] - expected_g) < 20, f"Frame 29 should be 40% red, 60% green"
    
    frame_30 = frames[29]
    avg_30 = np.mean(frame_30[:, 300:], axis=(0, 1))[:3]
    expected_r = 255 * 0.2
    expected_g = 255 * 0.8
    assert abs(avg_30[0] - expected_r) < 20 and abs(avg_30[1] - expected_g) < 20, f"Frame 30 should be 20% red, 80% green"
    
    avg_bg_31_52 = np.mean(frames_array[30:52, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_31_52[0] < 10 and abs(avg_bg_31_52[1] - 255) < 10 and avg_bg_31_52[2] < 10, f"Frames 31-52 should be green"
    
    frame_53 = frames[52]
    avg_53 = np.mean(frame_53[:, 300:], axis=(0, 1))[:3]
    expected_g = 255 * 0.8
    expected_b = 255 * 0.2
    assert abs(avg_53[0] - expected_g * 0) < 20 and abs(avg_53[1] - expected_g) < 20 and abs(avg_53[2] - expected_b) < 20, f"Frame 53 should be 80% green, 20% blue"
    
    frame_56 = frames[55]
    avg_56 = np.mean(frame_56[:, 300:], axis=(0, 1))[:3]
    expected_g = 255 * 0.2
    expected_b = 255 * 0.8
    assert abs(avg_56[0] - expected_g * 0) < 20 and abs(avg_56[1] - expected_g) < 20 and abs(avg_56[2] - expected_b) < 20, f"Frame 56 should be 20% green, 80% blue"
    
    avg_bg_57_82 = np.mean(frames_array[56:82, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_57_82[0] < 10 and avg_bg_57_82[1] < 10 and abs(avg_bg_57_82[2] - 255) < 10, f"Frames 57-82 should be blue"
    
    audio_220 = analyze_audio_frequency(audio_samples[0:80])
    assert 200 < audio_220 < 240, f"Audio start should be ~220Hz, got {audio_220}"
    
    audio_440 = analyze_audio_frequency(audio_samples[80:160])
    assert 420 < audio_440 < 460, f"Audio middle should be ~440Hz, got {audio_440}"
    
    audio_880 = analyze_audio_frequency(audio_samples[160:])
    assert 860 < audio_880 < 900, f"Audio end should be ~880Hz, got {audio_880}"
    
    for i in range(len(frames)):
        result = read_pixel_position(frames[i])
        
        if i < 26:
            expected_source, expected_frame = 0, i
        elif i < 52:
            expected_source, expected_frame = 1, i - 26
        else:
            expected_source, expected_frame = 2, i - 52
        
        actual_source = result[0] if result else None
        actual_frame = result[1] if result else None
        
        if result is None:
            continue
        
        assert actual_source == expected_source and actual_frame == expected_frame, \
            f"Frame {i}: expected source={expected_source}, frame={expected_frame}, got source={actual_source}, frame={actual_frame}"


def test_ease_in_out(source_videos, output_dir):
    """Test: ease_in_out blend mode"""
    videos = [
        str(source_videos["source1"]),
        str(source_videos["source2"]),
        str(source_videos["source3"]),
    ]
    output_path = os.path.join(output_dir, "combined-3x30i-4o-ease_in_out.mp4")
    
    generate_combined(videos, output_path, 4, BlendMode.EASE_IN_OUT, BlendMode.EASE_IN_OUT)
    
    frames, audio_samples, _, _ = get_frame_info(output_path)
    
    assert len(frames) == 82, f"Expected 82 frames, got {len(frames)}"
    
    frames_array = np.array(frames)
    
    avg_bg_1_26 = np.mean(frames_array[0:26, :, 300:], axis=(0, 1, 2))[:3]
    assert abs(avg_bg_1_26[0] - 255) < 10 and avg_bg_1_26[1] < 10 and avg_bg_1_26[2] < 10, f"Frames 1-26 should be red"
    
    expected_r_27 = 255 * 0.9045
    expected_g_27 = 255 * 0.0955
    frame_27 = frames[26]
    avg_27 = np.mean(frame_27[:, 300:], axis=(0, 1))[:3]
    assert abs(avg_27[0] - expected_r_27) < 20 and abs(avg_27[1] - expected_g_27) < 20, f"Frame 27 should be ~90.45% red, ~9.55% green"
    
    expected_r_28 = 255 * 0.6545
    expected_g_28 = 255 * 0.3455
    frame_28 = frames[27]
    avg_28 = np.mean(frame_28[:, 300:], axis=(0, 1))[:3]
    assert abs(avg_28[0] - expected_r_28) < 20 and abs(avg_28[1] - expected_g_28) < 20, f"Frame 28 should be ~65.45% red, ~34.55% green"
    
    expected_r_29 = 255 * 0.3455
    expected_g_29 = 255 * 0.6545
    frame_29 = frames[28]
    avg_29 = np.mean(frame_29[:, 300:], axis=(0, 1))[:3]
    assert abs(avg_29[0] - expected_r_29) < 20 and abs(avg_29[1] - expected_g_29) < 20, f"Frame 29 should be ~34.55% red, ~65.45% green"
    
    expected_r_30 = 255 * 0.0955
    expected_g_30 = 255 * 0.9045
    frame_30 = frames[29]
    avg_30 = np.mean(frame_30[:, 300:], axis=(0, 1))[:3]
    assert abs(avg_30[0] - expected_r_30) < 20 and abs(avg_30[1] - expected_g_30) < 20, f"Frame 30 should be ~9.55% red, ~90.45% green"
    
    avg_bg_31_52 = np.mean(frames_array[30:52, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_31_52[0] < 10 and abs(avg_bg_31_52[1] - 255) < 10 and avg_bg_31_52[2] < 10, f"Frames 31-52 should be green"
    
    avg_bg_57_82 = np.mean(frames_array[56:82, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_57_82[0] < 10 and avg_bg_57_82[1] < 10 and abs(avg_bg_57_82[2] - 255) < 10, f"Frames 57-82 should be blue"
    
    audio_220 = analyze_audio_frequency(audio_samples[0:80])
    assert 200 < audio_220 < 240, f"Audio start should be ~220Hz, got {audio_220}"
    
    audio_440 = analyze_audio_frequency(audio_samples[80:160])
    assert 420 < audio_440 < 460, f"Audio middle should be ~440Hz, got {audio_440}"
    
    audio_880 = analyze_audio_frequency(audio_samples[160:])
    assert 860 < audio_880 < 900, f"Audio end should be ~880Hz, got {audio_880}"
    
    for i in range(len(frames)):
        result = read_pixel_position(frames[i])
        
        if i < 26:
            expected_source, expected_frame = 0, i
        elif i < 52:
            expected_source, expected_frame = 1, i - 26
        else:
            expected_source, expected_frame = 2, i - 52
        
        actual_source = result[0] if result else None
        actual_frame = result[1] if result else None
        
        if result is None:
            continue
        
        assert actual_source == expected_source and actual_frame == expected_frame, \
            f"Frame {i}: expected source={expected_source}, frame={expected_frame}, got source={actual_source}, frame={actual_frame}"


def test_newer_only(source_videos, output_dir):
    """Test: newer_only blend mode"""
    videos = [
        str(source_videos["source1"]),
        str(source_videos["source2"]),
        str(source_videos["source3"]),
    ]
    output_path = os.path.join(output_dir, "combined-3x30i-4o-newer_only.mp4")
    
    generate_combined(videos, output_path, 4, BlendMode.NEWER_ONLY, BlendMode.NEWER_ONLY)
    
    frames, audio_samples, _, _ = get_frame_info(output_path)
    
    assert len(frames) == 82, f"Expected 82 frames, got {len(frames)}"
    
    frames_array = np.array(frames)
    
    avg_bg_0_25 = np.mean(frames_array[0:26, :, 300:], axis=(0, 1, 2))[:3]
    assert abs(avg_bg_0_25[0] - 255) < 10 and avg_bg_0_25[1] < 10 and avg_bg_0_25[2] < 10, f"Frames 0-25 should be red"
    
    avg_bg_30_51 = np.mean(frames_array[30:52, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_30_51[0] < 10 and abs(avg_bg_30_51[1] - 255) < 10 and avg_bg_30_51[2] < 10, f"Frames 30-51 should be green"
    
    avg_bg_60_81 = np.mean(frames_array[60:82, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_60_81[0] < 10 and avg_bg_60_81[1] < 10 and abs(avg_bg_60_81[2] - 255) < 10, f"Frames 60-81 should be blue"
    
    audio_220 = analyze_audio_frequency(audio_samples[0:87])
    assert 200 < audio_220 < 240, f"Audio frames 0-86 should be ~220Hz, got {audio_220}"
    
    audio_440 = analyze_audio_frequency(audio_samples[87:174])
    assert 420 < audio_440 < 460, f"Audio frames 87-173 should be ~440Hz, got {audio_440}"
    
    audio_880 = analyze_audio_frequency(audio_samples[174:])
    assert 860 < audio_880 < 900, f"Audio frames 174+ should be ~880Hz, got {audio_880}"
    
    for i in range(len(frames)):
        result = read_pixel_position(frames[i])
        
        if i < 26:
            expected_source, expected_frame = 0, i
        elif i < 52:
            expected_source, expected_frame = 1, i - 26
        else:
            expected_source, expected_frame = 2, i - 52
        
        if i in [26, 27, 28, 29, 52, 53, 54, 55, 56, 57, 58, 59]:
            continue
        
        actual_source = result[0] if result else None
        actual_frame = result[1] if result else None
        
        if result is None:
            continue
        
        assert actual_source == expected_source and actual_frame == expected_frame, \
            f"Frame {i}: expected source={expected_source}, frame={expected_frame}, got source={actual_source}, frame={actual_frame}"


def test_equal_power(source_videos, output_dir):
    """Test: equal_power blend mode"""
    videos = [
        str(source_videos["source1"]),
        str(source_videos["source2"]),
        str(source_videos["source3"]),
    ]
    output_path = os.path.join(output_dir, "combined-3x30i-4o-equal_power.mp4")
    
    generate_combined(videos, output_path, 4, BlendMode.EQUAL_POWER, BlendMode.EQUAL_POWER)
    
    frames, audio_samples, _, _ = get_frame_info(output_path)
    
    assert len(frames) == 82, f"Expected 82 frames, got {len(frames)}"
    
    frames_array = np.array(frames)
    
    avg_bg_1_26 = np.mean(frames_array[0:26, :, 300:], axis=(0, 1, 2))[:3]
    assert abs(avg_bg_1_26[0] - 255) < 10 and avg_bg_1_26[1] < 10 and avg_bg_1_26[2] < 10, f"Frames 1-26 should be red"
    
    frame_27 = frames[26]
    avg_27 = np.mean(frame_27[:, 300:], axis=(0, 1))[:3]
    expected_r = 255 * 0.691
    expected_g = 255 * 0.309
    assert abs(avg_27[0] - expected_r) < 20 and abs(avg_27[1] - expected_g) < 20, f"Frame 27 should be ~69.1% red, ~30.9% green"
    
    frame_28 = frames[27]
    avg_28 = np.mean(frame_28[:, 300:], axis=(0, 1))[:3]
    expected_r = 255 * 0.412
    expected_g = 255 * 0.588
    assert abs(avg_28[0] - expected_r) < 20 and abs(avg_28[1] - expected_g) < 20, f"Frame 28 should be ~41.2% red, ~58.8% green"
    
    frame_29 = frames[28]
    avg_29 = np.mean(frame_29[:, 300:], axis=(0, 1))[:3]
    expected_r = 255 * 0.191
    expected_g = 255 * 0.809
    assert abs(avg_29[0] - expected_r) < 20 and abs(avg_29[1] - expected_g) < 20, f"Frame 29 should be ~19.1% red, ~80.9% green"
    
    frame_30 = frames[29]
    avg_30 = np.mean(frame_30[:, 300:], axis=(0, 1))[:3]
    expected_r = 255 * 0.049
    expected_g = 255 * 0.951
    assert abs(avg_30[0] - expected_r) < 20 and abs(avg_30[1] - expected_g) < 20, f"Frame 30 should be ~4.9% red, ~95.1% green"
    
    avg_bg_31_52 = np.mean(frames_array[30:52, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_31_52[0] < 10 and abs(avg_bg_31_52[1] - 255) < 10 and avg_bg_31_52[2] < 10, f"Frames 31-52 should be green"
    
    avg_bg_57_82 = np.mean(frames_array[56:82, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_57_82[0] < 10 and avg_bg_57_82[1] < 10 and abs(avg_bg_57_82[2] - 255) < 10, f"Frames 57-82 should be blue"
    
    audio_220 = analyze_audio_frequency(audio_samples[0:80])
    assert 200 < audio_220 < 240, f"Audio start should be ~220Hz, got {audio_220}"
    
    audio_440 = analyze_audio_frequency(audio_samples[80:160])
    assert 420 < audio_440 < 460, f"Audio middle should be ~440Hz, got {audio_440}"
    
    audio_880 = analyze_audio_frequency(audio_samples[160:])
    assert 860 < audio_880 < 900, f"Audio end should be ~880Hz, got {audio_880}"
    
    for i in range(len(frames)):
        result = read_pixel_position(frames[i])
        
        if i < 26:
            expected_source, expected_frame = 0, i
        elif i < 52:
            expected_source, expected_frame = 1, i - 26
        else:
            expected_source, expected_frame = 2, i - 52
        
        actual_source = result[0] if result else None
        actual_frame = result[1] if result else None
        
        if result is None:
            continue
        
        assert actual_source == expected_source and actual_frame == expected_frame, \
            f"Frame {i}: expected source={expected_source}, frame={expected_frame}, got source={actual_source}, frame={actual_frame}"
