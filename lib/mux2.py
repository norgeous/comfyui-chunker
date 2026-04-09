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
        out_video_stream = output_container.add_stream("h264", rate=fps_int)
        out_video_stream.width = video_streams[0].width
        out_video_stream.height = video_streams[0].height
        out_video_stream.pix_fmt = video_streams[0].pix_fmt
        out_video_stream.options = {'preset': 'slow', 'crf': '10'}
        
        out_audio_stream = output_container.add_stream("aac", rate=audio_streams[0].rate)
        out_audio_stream.layout = audio_streams[0].layout
        
        video_packets_all = []
        for container, stream in zip(input_containers, video_streams):
            packets = list(container.demux(stream))
            video_packets_all.append(packets)
        
        video_packets = blend_all_packets(mode, video_packets_all, overlap_count)
        
        for packet in video_packets:
            packet.stream = out_video_stream
            output_container.mux(packet)
        
        for packet in out_video_stream.encode():
            packet.stream = out_video_stream
            output_container.mux(packet)
        
        audio_packets_all = []
        for container, stream in zip(input_containers, audio_streams):
            packets = list(container.demux(stream))
            audio_packets_all.append(packets)
        
        audio_packets = blend_all_packets(mode, audio_packets_all, overlap_count)
        
        for packet in audio_packets:
            packet.stream = out_audio_stream
            output_container.mux(packet)
        
        for packet in out_audio_stream.encode():
            packet.stream = out_audio_stream
            output_container.mux(packet)
    
    for c in input_containers:
        c.close()
