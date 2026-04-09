import os
import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import av
from lib.mux import mux

AUDIO_SAMPLE_RATE = 44100
AUDIO_DURATION = 2.0
VIDEO_FPS = 15
VIDEO_WIDTH = 512
VIDEO_HEIGHT = 512
VIDEO_FRAMES = 30


@pytest.fixture(scope="session")
def source_videos():
    """Generate source videos once per test session"""
    os.makedirs("test-source", exist_ok=True)
    
    sources = [
        ("source1.mp4", (255, 0, 0), 1, 400, 0, (255, 255, 255)),
        ("source2.mp4", (0, 255, 0), 2, 500, 10, (255, 255, 255)),
        ("source3.mp4", (0, 0, 255), 3, 600, 20, (255, 255, 255)),
    ]
    
    for filename, color, line_num, freq, pixel_y_offset, pixel_color in sources:
        path = os.path.join("test-source", filename)
        create_video_with_audio(path, color, line_num, freq, pixel_y_offset, pixel_color)
    
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


def create_video_with_audio(output_path, bg_color, line_num, freq, pixel_y_offset=0, pixel_color=(255, 255, 255)):
    from fractions import Fraction
    
    output = av.open(output_path, mode='w')
    video_stream = output.add_stream('h264', rate=VIDEO_FPS)
    video_stream.width = VIDEO_WIDTH
    video_stream.height = VIDEO_HEIGHT
    video_stream.pix_fmt = 'yuv420p'
    video_stream.options = {'preset': 'slow', 'crf': '10'}
    
    audio_stream = output.add_stream('aac', rate=AUDIO_SAMPLE_RATE)
    audio_stream.layout = 'stereo'
    
    num_samples = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION)
    t = np.linspace(0, AUDIO_DURATION, num_samples, endpoint=False)
    sine_wave = np.sin(2 * np.pi * freq * t) * 0.5
    
    chunk_size = 4096
    for chunk_start in range(0, num_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_samples)
        chunk = sine_wave[chunk_start:chunk_end]
        stereo_chunk = np.stack([chunk, chunk], axis=0).astype(np.float32)
        
        audio_frame = av.AudioFrame.from_ndarray(stereo_chunk, format='fltp', layout='stereo')
        audio_frame.pts = chunk_start
        audio_frame.time_base = Fraction(1, AUDIO_SAMPLE_RATE)
        audio_frame.sample_rate = AUDIO_SAMPLE_RATE
        
        for packet in audio_stream.encode(audio_frame):
            output.mux(packet)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        font = ImageFont.load_default()
    
    frame_duration = Fraction(1, VIDEO_FPS)
    
    for frame_num in range(VIDEO_FRAMES):
        img = Image.new('RGB', (VIDEO_WIDTH, VIDEO_HEIGHT), bg_color)
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([0, 0, 299, 29], fill=(0, 0, 0))
        
        pixel_x = frame_num * 10
        draw.rectangle([pixel_x, pixel_y_offset, pixel_x + 9, pixel_y_offset + 9], fill=pixel_color)
        
        y_pos = 50 + (line_num - 1) * 80
        draw.text((50, y_pos), str(frame_num + 1), fill=(255, 255, 255), font=font)
        
        frame = av.VideoFrame.from_ndarray(np.array(img), format='rgb24')
        frame = frame.reformat(format='yuv420p')
        frame.pts = frame_num
        frame.time_base = Fraction(1, VIDEO_FPS)
        
        for packet in video_stream.encode(frame):
            output.mux(packet)
    
    for packet in video_stream.encode():
        output.mux(packet)
    
    for packet in audio_stream.encode():
        output.mux(packet)
    
    output.close()


def get_frame_info(video_path):
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    audio_stream = container.streams.audio[0]
    
    frames = []
    for frame in container.decode(video_stream):
        img = frame.to_ndarray(format='rgb24')
        frames.append(img)
    
    container.close()
    
    container = av.open(video_path)
    audio_stream = container.streams.audio[0]
    
    audio_samples = []
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
    
    The pixel is a 10x10 colored square that moves 3px per frame.
    Source is determined by y-position of the pixel (0-2, 3-5, 6-8).
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
    """Test: No overlap (3 videos, 90 frames)"""
    videos = [
        str(source_videos["source1"]),
        str(source_videos["source2"]),
        str(source_videos["source3"]),
    ]
    output_path = os.path.join(output_dir, "mux-3x30i-0o-none.mp4")
    
    mux(videos, output_path, 0, "older_only")
    
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

    audio_400 = analyze_audio_frequency(audio_samples[0:87])
    assert 350 < audio_400 < 450, f"Audio frames 0-86 should be ~400Hz, got {audio_400}"
    
    audio_500 = analyze_audio_frequency(audio_samples[87:174])
    assert 450 < audio_500 < 550, f"Audio frames 87-173 should be ~500Hz, got {audio_500}"
    
    audio_600 = analyze_audio_frequency(audio_samples[174:])
    assert 550 < audio_600 < 650, f"Audio frames 174+ should be ~600Hz, got {audio_600}"


def test_older_only(source_videos, output_dir):
    """Test: older_only blend mode"""
    videos = [
        str(source_videos["source1"]),
        str(source_videos["source2"]),
        str(source_videos["source3"]),
    ]
    output_path = os.path.join(output_dir, "mux-3x30i-4o-older_only.mp4")
    
    mux(videos, output_path, 4, "older_only")
    
    frames, audio_samples, _, _ = get_frame_info(output_path)
    
    assert len(frames) == 82, f"Expected 82 frames, got {len(frames)}"
    
    frames_array = np.array(frames)
    
    avg_bg_1_30 = np.mean(frames_array[0:30, :, 300:], axis=(0, 1, 2))[:3]
    assert abs(avg_bg_1_30[0] - 255) < 10 and avg_bg_1_30[1] < 10 and avg_bg_1_30[2] < 10, f"Frames 1-30 should be red"
    
    avg_bg_31_56 = np.mean(frames_array[30:56, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_31_56[0] < 10 and abs(avg_bg_31_56[1] - 255) < 10 and avg_bg_31_56[2] < 10, f"Frames 31-56 should be green"
    
    avg_bg_57_82 = np.mean(frames_array[56:82, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_57_82[0] < 10 and avg_bg_57_82[1] < 10 and abs(avg_bg_57_82[2] - 255) < 10, f"Frames 57-82 should be blue"
    
    audio_400 = analyze_audio_frequency(audio_samples[0:87])
    assert 350 < audio_400 < 450, f"Audio frames 0-86 should be ~400Hz, got {audio_400}"
    
    audio_500 = analyze_audio_frequency(audio_samples[87:174])
    assert 450 < audio_500 < 550, f"Audio frames 87-173 should be ~500Hz, got {audio_500}"
    
    audio_600 = analyze_audio_frequency(audio_samples[174:])
    assert 550 < audio_600 < 650, f"Audio frames 174+ should be ~600Hz, got {audio_600}"
    
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
    """Test: linear_blend blend mode"""
    videos = [
        str(source_videos["source1"]),
        str(source_videos["source2"]),
        str(source_videos["source3"]),
    ]
    output_path = os.path.join(output_dir, "mux-3x30i-4o-linear_blend.mp4")
    
    mux(videos, output_path, 4, "linear_blend")
    
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
    
    audio_400 = analyze_audio_frequency(audio_samples[0:80])
    assert 350 < audio_400 < 450, f"Audio start should be ~400Hz, got {audio_400}"
    
    audio_500 = analyze_audio_frequency(audio_samples[80:160])
    assert 450 < audio_500 < 550, f"Audio middle should be ~500Hz, got {audio_500}"
    
    audio_600 = analyze_audio_frequency(audio_samples[160:])
    assert 550 < audio_600 < 650, f"Audio end should be ~600Hz, got {audio_600}"
    
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
    output_path = os.path.join(output_dir, "mux-3x30i-4o-ease_in_out.mp4")
    
    mux(videos, output_path, 4, "ease_in_out")
    
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
    
    audio_400 = analyze_audio_frequency(audio_samples[0:80])
    assert 350 < audio_400 < 450, f"Audio start should be ~400Hz, got {audio_400}"
    
    audio_500 = analyze_audio_frequency(audio_samples[80:160])
    assert 450 < audio_500 < 550, f"Audio middle should be ~500Hz, got {audio_500}"
    
    audio_600 = analyze_audio_frequency(audio_samples[160:])
    assert 550 < audio_600 < 650, f"Audio end should be ~600Hz, got {audio_600}"
    
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
    output_path = os.path.join(output_dir, "mux-3x30i-4o-newer_only.mp4")
    
    mux(videos, output_path, 4, "newer_only")
    
    frames, audio_samples, _, _ = get_frame_info(output_path)
    
    assert len(frames) == 82, f"Expected 82 frames, got {len(frames)}"
    
    frames_array = np.array(frames)
    
    avg_bg_0_25 = np.mean(frames_array[0:26, :, 300:], axis=(0, 1, 2))[:3]
    assert abs(avg_bg_0_25[0] - 255) < 10 and avg_bg_0_25[1] < 10 and avg_bg_0_25[2] < 10, f"Frames 0-25 should be red"
    
    avg_bg_30_51 = np.mean(frames_array[30:52, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_30_51[0] < 10 and abs(avg_bg_30_51[1] - 255) < 10 and avg_bg_30_51[2] < 10, f"Frames 30-51 should be green"
    
    avg_bg_60_81 = np.mean(frames_array[60:82, :, 300:], axis=(0, 1, 2))[:3]
    assert avg_bg_60_81[0] < 10 and avg_bg_60_81[1] < 10 and abs(avg_bg_60_81[2] - 255) < 10, f"Frames 60-81 should be blue"
    
    audio_400 = analyze_audio_frequency(audio_samples[0:87])
    assert 350 < audio_400 < 450, f"Audio frames 0-86 should be ~400Hz, got {audio_400}"
    
    audio_500 = analyze_audio_frequency(audio_samples[87:174])
    assert 450 < audio_500 < 550, f"Audio frames 87-173 should be ~500Hz, got {audio_500}"
    
    audio_600 = analyze_audio_frequency(audio_samples[174:])
    assert 550 < audio_600 < 650, f"Audio frames 174+ should be ~600Hz, got {audio_600}"
    
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
