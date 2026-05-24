import math

def plan_chunks(length_adjuster, chunk_length: int, chunk_overlap: int, total_length: int) -> tuple[list[int], int, int]:
    adjusted_chunk_length = length_adjuster(chunk_length)
    stride = adjusted_chunk_length - chunk_overlap
    full_chunk_count = max(0, math.ceil((total_length - adjusted_chunk_length) / stride))
    chunk_lengths = [adjusted_chunk_length] * full_chunk_count
    pos = full_chunk_count * stride
    if pos < total_length:
        tail = total_length - pos
        chunk_lengths.append(length_adjuster(tail))
    adjusted_total_length = sum(chunk_lengths) - chunk_overlap * (len(chunk_lengths) - 1)
    return adjusted_chunk_length, adjusted_total_length, chunk_lengths
