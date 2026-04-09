import av
from fractions import Fraction
from .utils_blend_mode import OverlapBlendMode
from .utils_blend_packets import blend_all_packets


def mux(videos: list, output_path: str, overlap_count: int, overlap_blend_mode: str):
    if not videos:
        raise ValueError("At least one video required")
    
    if overlap_count < 0:
        raise ValueError("overlap_count must be 0 or greater")
    
    mode = OverlapBlendMode(overlap_blend_mode)
    
    # Open all input containers
    input_containers = [av.open(v) for v in videos]
    
    video_streams = [c.streams.video[0] for c in input_containers]
    audio_streams = [c.streams.audio[0] for c in input_containers]
    
    with av.open(output_path, mode="w", format="mp4") as output_container:
        # Use template-based stream creation for proper remuxing
        out_video_stream = output_container.add_stream_from_template(video_streams[0])
        out_video_stream.options = {'preset': 'slow', 'crf': '10'}
        # Set time_base to match input to avoid timing issues
        out_video_stream.time_base = video_streams[0].time_base
        out_video_stream.codec_context.time_base = video_streams[0].time_base
        
        out_audio_stream = output_container.add_stream_from_template(audio_streams[0])
        out_audio_stream.time_base = audio_streams[0].time_base
        
        # Demux video packets from each source
        video_packets_all = []
        
        for container, video_stream in zip(input_containers, video_streams):
            video_packets = []
            
            for packet in container.demux(video_stream):
                if packet.dts is None:
                    continue
                video_packets.append(packet)
            
            video_packets_all.append(video_packets)
        
        # Process video: handle overlap blending
        video_pts_counter = [0]
        video_packets = blend_all_packets(
            mode, video_packets_all, overlap_count,
            out_video_stream, video_pts_counter
        )
        
        # Mux video packets
        for packet in video_packets:
            packet.stream = out_video_stream
            output_container.mux(packet)
        
        # Flush video encoder
        for packet in out_video_stream.encode():
            packet.stream = out_video_stream
            output_container.mux(packet)
        
        # Audio: demux and remux with rebased timestamps
        audio_pts = 0
        for video_path in videos:
            audio_container = av.open(video_path)
            audio_stream = audio_container.streams.audio[0]
            
            for packet in audio_container.demux(audio_stream):
                if packet.dts is None:
                    continue
                packet.stream = out_audio_stream
                packet.pts = audio_pts
                packet.dts = audio_pts
                output_container.mux(packet)
                audio_pts += 1
            
            audio_container.close()
    
    for c in input_containers:
        c.close()