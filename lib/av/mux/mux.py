import av
import numpy as np
from typing import List, Any
from fractions import Fraction
from ...utils_blend_mode import get_blend_factor, OverlapBlendMode

SAMPLES_PER_AUDIO_FRAME = 1024


def calculate_output_frame_counts(
    videos: List[str],
    video_frame_counts: List[int],
    overlap_count: int,
    overlap_blend_mode: str,
) -> List[int]:
    num_videos = len(videos)
    output_counts = []
    
    for video_idx in range(num_videos):
        if overlap_count == 0:
            max_frames = video_frame_counts[video_idx]
        elif video_idx == 0:
            max_frames = video_frame_counts[video_idx]
            if overlap_blend_mode in ["linear_blend", "ease_in_out"]:
                max_frames = video_frame_counts[video_idx] - overlap_count
            elif overlap_blend_mode == "newer_only":
                max_frames = video_frame_counts[video_idx] - overlap_count
        elif overlap_blend_mode in ["linear_blend", "ease_in_out"]:
            max_frames = video_frame_counts[video_idx]
            if video_idx < num_videos - 1:
                max_frames = video_frame_counts[video_idx] - overlap_count
        elif overlap_blend_mode == "newer_only":
            if video_idx == num_videos - 1:
                max_frames = video_frame_counts[video_idx]
            else:
                max_frames = video_frame_counts[video_idx] - overlap_count
        else:
            max_frames = video_frame_counts[video_idx]
        
        if overlap_blend_mode == "newer_only" and video_idx == num_videos - 1:
            max_frames = video_frame_counts[video_idx]
        
        output_counts.append(max_frames)
    
    return output_counts


def process_video_stream(
    input_containers: List[Any],
    video_frames_all: List[List[Any]],
    video_frame_counts: List[int],
    output_container: Any,
    out_video_stream: Any,
    fps_int: int,
    overlap_count: int,
    overlap_blend_mode: str,
    output_frame_counts: List[int],
) -> int:
    frame_duration = Fraction(1, fps_int)
    output_frame_num = 0
    num_videos = len(input_containers)
    
    for video_idx in range(num_videos):
        max_frames = output_frame_counts[video_idx]
        
        if overlap_count == 0:
            frame_start = 0
            blend_range = 0
        elif video_idx == 0:
            frame_start = 0
            blend_range = 0
            if overlap_blend_mode in ["linear_blend", "ease_in_out"]:
                max_frames = video_frame_counts[video_idx] - overlap_count
            elif overlap_blend_mode == "newer_only":
                max_frames = video_frame_counts[video_idx] - overlap_count
        elif overlap_blend_mode in ["linear_blend", "ease_in_out"]:
            frame_start = 0
            blend_range = overlap_count
            if video_idx < num_videos - 1:
                max_frames = video_frame_counts[video_idx] - overlap_count
        elif overlap_blend_mode == "newer_only":
            if video_idx == num_videos - 1:
                frame_start = 0
                max_frames = video_frame_counts[video_idx]
            else:
                frame_start = 0
                max_frames = video_frame_counts[video_idx] - overlap_count
            blend_range = overlap_count
        else:
            frame_start = overlap_count
            blend_range = overlap_count
        
        if overlap_blend_mode == "newer_only" and video_idx == num_videos - 1:
            max_frames = video_frame_counts[video_idx]
        
        for frame_idx in range(frame_start, max_frames):
            if frame_idx >= len(video_frames_all[video_idx]):
                break
            
            frame = video_frames_all[video_idx][frame_idx]
            
            in_blend = False
            if video_idx > 0 and overlap_count > 0:
                if overlap_blend_mode in ["linear_blend", "ease_in_out"]:
                    in_blend = frame_idx < blend_range
                elif overlap_blend_mode == "newer_only":
                    if video_idx < num_videos - 1:
                        in_blend = frame_idx >= video_frame_counts[video_idx] - overlap_count
                else:
                    in_blend = frame_idx < blend_range
            
            if in_blend:
                if overlap_blend_mode == "newer_only":
                    idx_in_blend = frame_idx - (video_frame_counts[video_idx] - overlap_count)
                else:
                    idx_in_blend = frame_idx
                
                neighbor_idx = video_idx - 1
                neighbor_frame_idx = (video_frame_counts[neighbor_idx] - overlap_count) + idx_in_blend
                neighbor_frame_idx = max(0, min(neighbor_frame_idx, len(video_frames_all[neighbor_idx]) - 1))
                neighbor_frame = video_frames_all[neighbor_idx][neighbor_frame_idx]
                
                # factor = get_blend_factor(overlap_blend_mode, idx_in_blend, overlap_count)
                factor = get_blend_factor(OverlapBlendMode(overlap_blend_mode), idx_in_blend / overlap_count)
                
                img1 = neighbor_frame.to_ndarray(format='rgb24')
                img2 = frame.to_ndarray(format='rgb24')
                blended = (img1 * (1 - factor) + img2 * factor).astype(np.uint8)
                
                out_frame = av.VideoFrame.from_ndarray(blended, format='rgb24')
                out_frame = out_frame.reformat(format='yuv420p')
                out_frame.pts = output_frame_num
                out_frame.time_base = frame_duration
                
                for pkt in out_video_stream.encode(out_frame):
                    output_container.mux(pkt)
            else:
                frame.pts = output_frame_num
                frame.time_base = frame_duration
                for pkt in out_video_stream.encode(frame):
                    output_container.mux(pkt)
            
            output_frame_num += 1
    
    for pkt in out_video_stream.encode():
        output_container.mux(pkt)
    
    return output_frame_num


def process_audio_stream(
    input_containers: List[Any],
    audio_frames_all: List[List[Any]],
    audio_frame_counts: List[int],
    output_container: Any,
    out_audio_stream: Any,
    audio_sample_rate: int,
    fps_int: int,
    overlap_count: int,
    overlap_blend_mode: str,
    output_frame_counts: List[int],
) -> int:
    audio_frame_duration = Fraction(1, audio_sample_rate)
    audio_samples_per_frame = audio_sample_rate // fps_int
    
    output_audio_sample = 0
    num_videos = len(input_containers)
    
    for video_idx in range(num_videos):
        video_start_output = sum(output_frame_counts[:video_idx]) if video_idx > 0 else 0
        
        audio_source_start = 0
        if overlap_count > 0 and video_idx > 0:
            audio_overlap_frames = round(overlap_count * audio_samples_per_frame / SAMPLES_PER_AUDIO_FRAME)
            if overlap_blend_mode not in ["linear_blend", "ease_in_out"]:
                audio_source_start = audio_overlap_frames
        
        video_frames_this = output_frame_counts[video_idx]
        audio_frames_needed = round(video_frames_this * audio_samples_per_frame / SAMPLES_PER_AUDIO_FRAME)
        
        audio_source_frames = audio_frame_counts[video_idx] - audio_source_start
        frames_to_output = min(audio_frames_needed, audio_source_frames)
        
        for i in range(frames_to_output):
            frame_idx = audio_source_start + i
            if frame_idx >= audio_frame_counts[video_idx]:
                break
            frame = audio_frames_all[video_idx][frame_idx]
            
            video_current_frame = int(output_audio_sample * fps_int / audio_sample_rate)
            
            audio_overlap_frames = round(overlap_count * audio_samples_per_frame / SAMPLES_PER_AUDIO_FRAME)
            
            in_blend = False
            video_frames_since_start = 0
            if video_idx > 0 and overlap_count > 0:
                if overlap_blend_mode in ["linear_blend", "ease_in_out"]:
                    video_frames_since_start = video_current_frame - video_start_output
                    in_blend = 0 <= video_frames_since_start < overlap_count + 2
                elif overlap_blend_mode == "newer_only":
                    video_frames_since_start = video_current_frame - video_start_output
                    video_total = output_frame_counts[video_idx]
                    in_blend = video_frames_since_start >= video_total - overlap_count
                else:
                    video_frames_since_start = video_current_frame - video_start_output
                    in_blend = 0 <= video_frames_since_start < overlap_count
            
            if in_blend:
                neighbor_idx = video_idx - 1
                neighbor_frame_idx = audio_frame_counts[neighbor_idx] - audio_overlap_frames + i
                
                if 0 <= neighbor_frame_idx < len(audio_frames_all[neighbor_idx]):
                    neighbor_frame = audio_frames_all[neighbor_idx][neighbor_frame_idx]
                    
                    idx_in_blend = video_frames_since_start
                    factor = get_blend_factor(OverlapBlendMode(overlap_blend_mode), idx_in_blend / overlap_count)
                    
                    samples_current = frame.to_ndarray()
                    samples_neighbor = neighbor_frame.to_ndarray()
                    min_samples = min(samples_current.shape[1], samples_neighbor.shape[1])
                    blended = np.zeros((2, min_samples), dtype=np.float32)
                    blended[:, :min_samples] = (
                        samples_neighbor[:, :min_samples] * (1 - factor) +
                        samples_current[:, :min_samples] * factor
                    )
                    
                    out_frame = av.AudioFrame.from_ndarray(blended, format='fltp', layout='stereo')
                    out_frame.pts = output_audio_sample
                    out_frame.time_base = audio_frame_duration
                    out_frame.sample_rate = audio_sample_rate
                    
                    for pkt in out_audio_stream.encode(out_frame):
                        output_container.mux(pkt)
                else:
                    frame.pts = output_audio_sample
                    frame.time_base = audio_frame_duration
                    for pkt in out_audio_stream.encode(frame):
                        output_container.mux(pkt)
            else:
                frame.pts = output_audio_sample
                frame.time_base = audio_frame_duration
                for pkt in out_audio_stream.encode(frame):
                    output_container.mux(pkt)
            
            output_audio_sample += frame.samples
    
    for pkt in out_audio_stream.encode():
        output_container.mux(pkt)
    
    return output_audio_sample


def mux(paths: List[str], out_path: str, overlap_count: int, overlap_blend_mode: str):
    valid_modes = {"older_only", "linear_blend", "ease_in_out", "newer_only"}
    if overlap_blend_mode not in valid_modes:
        raise ValueError(f"Unknown blend mode: {overlap_blend_mode}")
    
    input_containers = [av.open(v) for v in paths]
    
    try:
        video_streams = [c.streams.video[0] for c in input_containers]
        audio_streams = [c.streams.audio[0] for c in input_containers]
        output_container = av.open(out_path, mode="w", format="mp4")
        
        try:
            fps = video_streams[0].average_rate
            fps_int = int(fps) if fps else 15
            
            out_video_stream = output_container.add_stream("h264", rate=fps_int)
            out_video_stream.width = video_streams[0].width
            out_video_stream.height = video_streams[0].height
            out_video_stream.pix_fmt = video_streams[0].pix_fmt
            
            out_audio_stream = output_container.add_stream("aac", rate=audio_streams[0].rate)
            out_audio_stream.layout = audio_streams[0].layout
            
            video_frames_all = [list(c.decode(video_streams[i])) for i, c in enumerate(input_containers)]
            video_frame_counts = [len(frames) for frames in video_frames_all]
            
            input_containers = [av.open(v) for v in paths]
            audio_streams = [c.streams.audio[0] for c in input_containers]
            audio_frames_all = [list(c.decode(audio_streams[i])) for i, c in enumerate(input_containers)]
            audio_frame_counts = [len(frames) for frames in audio_frames_all]
            
            output_frame_counts = calculate_output_frame_counts(
                paths, video_frame_counts, overlap_count, overlap_blend_mode
            )
            
            process_video_stream(
                input_containers, video_frames_all, video_frame_counts,
                output_container, out_video_stream, fps_int,
                overlap_count, overlap_blend_mode, output_frame_counts
            )
            
            audio_sample_rate = audio_streams[0].rate
            process_audio_stream(
                input_containers, audio_frames_all, audio_frame_counts,
                output_container, out_audio_stream, audio_sample_rate, fps_int,
                overlap_count, overlap_blend_mode, output_frame_counts
            )
            
            output_container.close()
        except Exception:
            output_container.close()
            raise
    finally:
        for c in input_containers:
            c.close()
