import av
import sys
import numpy as np
from fractions import Fraction
from .utils_blend_mode import get_blend_factor, OverlapBlendMode


def blend_packets(
    old_packet: av.Packet,
    new_packet: av.Packet,
    factor: float,
    out_video_stream,
    output_pts: int
) -> list[av.Packet]:
    old_frames = old_packet.decode()
    new_frames = new_packet.decode()
    
    # If either packet fails to decode, fall back to remuxing old packet
    if not old_frames or not new_frames:
        old_packet.pts = output_pts
        old_packet.dts = output_pts
        # Preserve original time_base so it can be remuxed correctly
        old_packet.time_base = old_packet.stream.time_base
        return [old_packet]
    
    old_frame = old_frames[0]
    new_frame = new_frames[0]
    
    img_old = old_frame.to_ndarray(format='rgb24')
    img_new = new_frame.to_ndarray(format='rgb24')
    
    blended = (img_old * (1 - factor) + img_new * factor).astype(np.uint8)
    
    out_frame = av.VideoFrame.from_ndarray(blended, format='rgb24')
    out_frame = out_frame.reformat(
        width=out_video_stream.width,
        height=out_video_stream.height,
        format='yuv420p'
    )
    out_frame.pts = output_pts
    out_frame.time_base = Fraction(1, out_video_stream.rate)
    
    packets = list(out_video_stream.encode(out_frame))
    return packets


def blend_all_packets(
    mode: OverlapBlendMode,
    all_packets: list[list[av.Packet]],
    overlap_count: int,
    out_video_stream,
    pts_counter: list[int]
) -> list[av.Packet]:
    # Calculate frame duration in time_base units
    # For 15fps with time_base 1/15360: each frame = 1024 time units
    rate = out_video_stream.average_rate
    time_base = out_video_stream.time_base
    
    # Default frame duration (assuming 15fps at 1/15360 time_base)
    frame_duration = 1024
    
    # Handle case where average_rate might be set from template
    if rate is not None:
        try:
            if hasattr(rate, 'numerator'):
                # It's a Fraction
                frame_duration = int(rate.numerator * time_base.denominator / rate.denominator / time_base.numerator)
            else:
                # It's a number
                frame_duration = int(time_base.denominator / rate / time_base.numerator)
        except (TypeError, ZeroDivisionError):
            pass  # Use default frame_duration
    
    out_packets = []
    for i, packets in enumerate(all_packets):
        is_first = i == 0
        is_last = i == len(all_packets) - 1
        for j, packet in enumerate(packets):
            if isinstance(packet.stream, av.video.VideoStream):

                # skip packets that overlap at start of clip (they're already used in blending)
                if j < overlap_count and not is_first:
                    continue

                # for packets that overlap at end of clip
                elif j > len(packets) - 1 - overlap_count and not is_last:
                    offset = j - (len(packets) - 1 - overlap_count)
                    other_packet = all_packets[i+1][offset]
                    factor = get_blend_factor(mode, j / overlap_count)
                    blended_packets = blend_packets(packet, other_packet, factor, out_video_stream, pts_counter[0] * frame_duration)
                    
                    # For blended packets, need to ensure is_keyframe is set on first packet of each new source
                    # Check if this is the first blended packet (at start of new source)
                    # The first packet of source i+1 (j=0) should become keyframe
                    if offset == 0:  # This is blending with first packet of next source
                        # The first packet after blending should be keyframe
                        for bp in blended_packets:
                            if bp.pts == pts_counter[0] * frame_duration:
                                # Check if new packet has keyframe
                                if other_packet.is_keyframe:
                                    # Force is_keyframe by creating a new packet property
                                    # Actually, let's just check what we get
                                    pass
                    
                    for bp in blended_packets:
                        out_packets.append(bp)
                    pts_counter[0] += 1
                
                # packets not inside the overlap
                else:
                    packet.stream = out_video_stream
                    packet.pts = pts_counter[0] * frame_duration
                    packet.dts = pts_counter[0] * frame_duration
                    out_packets.append(packet)
                    pts_counter[0] += 1
                
            if isinstance(packet.stream, av.audio.AudioStream):
                pass

    return out_packets