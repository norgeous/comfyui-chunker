import av
from fractions import Fraction

def combine_videos(input_paths, output_path, include_audio=True):
    """Combine multiple videos into one by muxing packets directly."""
    if not input_paths:
        raise ValueError("No input videos provided")

    inputs = [av.open(p) for p in input_paths]
    output = av.open(output_path, "w")

    in_video = inputs[0].streams.video[0]
    out_video = output.add_stream_from_template(in_video)
    out_video.time_base = in_video.time_base

    fps = in_video.average_rate
    video_time_base = in_video.time_base
    video_step = int(1 / Fraction(video_time_base) / fps)

    out_audio = None
    if include_audio:
        audio_streams = inputs[0].streams.audio
        if len(audio_streams) > 0:
            in_audio = audio_streams[0]
            out_audio = output.add_stream_from_template(in_audio)
            out_audio.time_base = in_audio.time_base

    # Collect all packets first
    all_data = []
    for inp in inputs:
        in_video = inp.streams.video[0]
        in_audio = None
        if include_audio and len(inp.streams.audio) > 0:
            in_audio = inp.streams.audio[0]
        
        in_streams = [in_video]
        if in_audio:
            in_streams.append(in_audio)
        
        v_pkts = []
        a_pkts = []
        
        for pkt in inp.demux(*in_streams):
            if pkt.stream == in_video:
                v_pkts.append(pkt)
            elif in_audio and pkt.stream == in_audio:
                a_pkts.append(pkt)
        
        all_data.append((v_pkts, a_pkts))
        inp.close()

    # Now mux with proper interleaving
    video_offset = 0
    audio_offset = 0
    
    for v_pkts, a_pkts in all_data:
        last_video_pts = 0
        
        # Mux video first
        for pkt in v_pkts:
            if pkt.dts is None:
                continue
            if pkt.pts is not None:
                last_video_pts = max(last_video_pts, pkt.pts)
                pkt.pts += video_offset
                pkt.dts += video_offset
            pkt.stream = out_video
            output.mux(pkt)

        if last_video_pts > 0:
            video_offset += last_video_pts + video_step
        elif video_offset == 0:
            video_offset = video_step

        # Mux audio with audio-specific offset
        if out_audio and a_pkts:
            for pkt in a_pkts:
                if pkt.size <= 0:
                    continue
                # Use audio_offset which accumulates separately like video
                pkt.pts = audio_offset
                pkt.dts = audio_offset
                pkt.stream = out_audio
                output.mux(pkt)
                audio_offset += video_step

    output.close()

    return output_path


if __name__ == "__main__":
    combine_videos(["source1.mp4", "source2.mp4"], "combined1+2.mp4")
    combine_videos(["source1.mp4", "source2.mp4", "source3.mp4"], "combined1+2+3.mp4")
    print("Done! Output written to combined.mp4")