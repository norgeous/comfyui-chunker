import math

def plan_chunks(length_adjuster, chunk_length: int, chunk_overlap: int, total: int) -> list[int]:
    actual = length_adjuster(chunk_length)
    stride = actual - chunk_overlap
    full_chunk_count = max(0, math.ceil((total - actual) / stride))
    chunks = [actual] * full_chunk_count
    pos = full_chunk_count * stride
    if pos >= total: return chunks
    tail = total - pos
    chunks.append(length_adjuster(tail))
    return chunks
