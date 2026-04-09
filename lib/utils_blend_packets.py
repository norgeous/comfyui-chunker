import av
from .utils_blend_mode import get_blend_factor, OverlapBlendMode


def blend_all_packets(mode: OverlapBlendMode, all_packets: list[list[av.Packet]], overlap_count=10) -> list[av.Packet]:
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
                    blended_packet = blend_packets(packet, other_packet, factor)
                    out_packets.append(blended_packet)
                
                # packets not inside the overlap
                else:
                    out_packets.append(packet)
                
            if isinstance(packet.stream, av.audio.AudioStream):
                pass

    return out_packets


def blend_packets(old_packet: av.Packet, new_packet: av.Packet, factor: float) -> av.Packet:
    old_data = old_packet.to_bytes()
    new_data = new_packet.to_bytes()
    
    max_len = max(len(old_data), len(new_data))
    old_bytes = old_data.ljust(max_len, b'\x00')
    new_bytes = new_data.ljust(max_len, b'\x00')
    
    blended = bytes(
        int(old_bytes[b] * (1 - factor) + new_bytes[b] * factor)
        for b in range(max_len)
    )
    
    new_pkt = av.Packet(blended)
    new_pkt.stream = old_packet.stream
    new_pkt.pts = old_packet.pts
    new_pkt.dts = old_packet.dts
    new_pkt.time_base = old_packet.time_base
    
    return new_pkt