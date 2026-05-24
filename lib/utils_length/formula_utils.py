import math

def plan_chunks(length_adjuster, chunk_length: int, chunk_overlap: int, total_length: int) -> tuple[list[int], int]:
    actual = length_adjuster(chunk_length)
    stride = actual - chunk_overlap
    full_chunk_count = max(0, math.ceil((total_length - actual) / stride))
    chunk_lengths = [actual] * full_chunk_count
    pos = full_chunk_count * stride
    if pos >= total_length: return chunk_lengths, sum(chunk_lengths) - chunk_overlap * (len(chunk_lengths) - 1)
    tail = total_length - pos
    chunk_lengths.append(length_adjuster(tail))
    return chunk_lengths, sum(chunk_lengths) - chunk_overlap * (len(chunk_lengths) - 1)
