from av import Packet
from .utils_blend_mode import get_blend_factor, OverlapBlendMode


def blend_videos(mode: OverlapBlendMode, all_packets: list[list[Packet]], overlap_count=10) -> list[Packet]:
    if not all_packets:
        return []
    
    if len(all_packets) == 1:
        return all_packets[0]
    
    result = list(all_packets[0])
    
    for i in range(1, len(all_packets)):
        segment = all_packets[i]
        if not segment:
            continue
        
        overlap_start = len(result) - overlap_count
        if overlap_start < 0:
            overlap_start = 0
        
        for j, packet in enumerate(segment):
            if j < overlap_count and overlap_start + j < len(result):
                old_packet = result[overlap_start + j]
                blend_percent = j / overlap_count
                factor = get_blend_factor(mode, blend_percent)
                packet = blend_packets(old_packet, packet, factor)
                result[overlap_start + j] = packet
            else:
                result.append(packet)
    
    return result


def blend_packets(old_packet: Packet, new_packet: Packet, factor: float) -> Packet:
    old_data = old_packet.to_bytes()
    new_data = new_packet.to_bytes()
    
    max_len = max(len(old_data), len(new_data))
    old_bytes = old_data.ljust(max_len, b'\x00')
    new_bytes = new_data.ljust(max_len, b'\x00')
    
    blended = bytes(
        int(old_bytes[b] * (1 - factor) + new_bytes[b] * factor)
        for b in range(max_len)
    )
    
    new_pkt = Packet(blended)
    new_pkt.stream = old_packet.stream
    new_pkt.pts = old_packet.pts
    new_pkt.dts = old_packet.dts
    new_pkt.time_base = old_packet.time_base
    
    return new_pkt