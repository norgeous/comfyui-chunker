import math
from typing import Callable, List, Tuple


def plan_chunks(length_adjuster: Callable[[int], int],
                chunk_length: int,
                overlap_length: int,
                total_length: int) -> Tuple[int,
                                            int,
                                            List[int]]:
    adjusted_chunk_length = length_adjuster(chunk_length)
    stride = adjusted_chunk_length - overlap_length
    full_chunk_count = max(
        0, math.ceil(
            (total_length - adjusted_chunk_length) / stride))
    chunk_lengths = [adjusted_chunk_length] * full_chunk_count
    pos = full_chunk_count * stride
    if pos < total_length:
        tail = total_length - pos
        chunk_lengths.append(length_adjuster(tail))
    adjusted_total_length = sum(chunk_lengths) - \
        overlap_length * (len(chunk_lengths) - 1)
    return adjusted_chunk_length, adjusted_total_length, chunk_lengths
