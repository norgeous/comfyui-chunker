import av
from .utils_blend_mode import OverlapBlendMode
from .utils_blend_packets import blend_all_packets


def mux(videos: list, output_path: str, overlap_count: int, overlap_blend_mode: str):
    if not videos:
        raise ValueError("At least one video required")
    
    if overlap_count < 0:
        raise ValueError("overlap_count must be 0 or greater")
    
    mode = OverlapBlendMode(overlap_blend_mode)
    
    input_containers = [av.open(v) for v in videos]
    
    video_streams = [c.streams.video[0] for c in input_containers]
    audio_streams = [c.streams.audio[0] for c in input_containers]
    
    fps = video_streams[0].average_rate
    fps_int = int(fps) if fps else 15
    
    with av.open(output_path, mode="w", format="mp4") as output_container:
        # Use template-based stream creation for proper remuxing
        out_video_stream = output_container.add_stream_from_template(video_streams[0])
        out_video_stream.options = {'preset': 'slow', 'crf': '10'}
        
        out_audio_stream = output_container.add_stream_from_template(audio_streams[0])
        
        # Demux all streams at once using tuple-based demux to avoid corruption
        video_packets_all = []
        audio_packets_all = []
        
        for i, (container, video_stream, audio_stream) in enumerate(zip(input_containers, video_streams, audio_streams)):
            video_packets = []
            audio_packets = []
            
            # Demux all streams at once - avoids demux corruption
            for packet in container.demux((video_stream, audio_stream)):
                if packet.dts is None:  # Skip flushing packets
                    continue
                if packet.stream == video_stream:
                    video_packets.append(packet)
                elif packet.stream == audio_stream:
                    audio_packets.append(packet)
            
            video_packets_all.append(video_packets)
            audio_packets_all.append(audio_packets)
        
        # Video processing: hybrid approach
        # - Non-overlapping: remux packets directly
        # - Overlapping: decode -> blend -> re-encode
        video_pts_counter = [0]
        video_packets = blend_all_packets(
            mode, video_packets_all, overlap_count, 
            out_video_stream, video_pts_counter
        )
        
        # Mux video packets
        for packet in video_packets:
            packet.stream = out_video_stream
            output_container.mux(packet)
        
        # Flush video encoder for any remaining packets from blending
        for packet in out_video_stream.encode():
            packet.stream = out_video_stream
            output_container.mux(packet)
        
        # Audio: remux directly (no blending)
        audio_pts = 0
        for packet_list in audio_packets_all:
            for packet in packet_list:
                if packet.dts is None:
                    continue
                packet.stream = out_audio_stream
                output_container.mux(packet)
                audio_pts += 1
        
        # Flush audio encoder
        for packet in out_audio_stream.encode():
            packet.stream = out_audio_stream
            output_container.mux(packet)
    
    for c in input_containers:
        c.close()