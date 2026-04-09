import av
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
    
    if not old_frames or not new_frames:
        return []
    
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
    out_packets = []
    for i, packets in enumerate(all_packets):
        is_first = i == 0
        is_last = i == len(all_packets) - 1
        for j, packet in enumerate(packets):
            if isinstance(packet.stream, av.video.VideoStream):

                # skip packets that overlap at start of clip (they're already used)
                if j < overlap_count and not is_first:
                    continue

                # for packets that overlap at end of clip
                elif j > len(packets) - 1 - overlap_count and not is_last:
                    offset = j - (len(packets) - 1 - overlap_count)
                    other_packet = all_packets[i+1][offset]
                    factor = get_blend_factor(mode, j / overlap_count)
                    blended_packets = blend_packets(packet, other_packet, factor, out_video_stream, pts_counter[0])
                    for bp in blended_packets:
                        out_packets.append(bp)
                    pts_counter[0] += 1
                
                # packets not inside the overlap
                else:
                    packet.stream = out_video_stream
                    packet.pts = pts_counter[0]
                    packet.dts = pts_counter[0]
                    out_packets.append(packet)
                    pts_counter[0] += 1
                
            if isinstance(packet.stream, av.audio.AudioStream):
                pass

    return out_packets