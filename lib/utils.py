from functools import reduce
import math

def log(*args, **kwargs):
    print(f"\U0001F36B  Chunker:", *args, **kwargs)

def count(list):
    if len(list) == 0: return 0
    if len(list) == 1: return len(list[0])
    return reduce(lambda acc, item: acc + len(item), [0, *list])

# force 4n+1 chunk_length. example: 1, 5, 9, 13, 17
def force_wan_length(value):
    return (math.ceil((value - 1) / 4) * 4) + 1

# force 8n+1 chunk_length. example: 1, 9, 17, 25, 33
def force_ltx2_length(value):
    return (math.ceil((value - 1) / 8) * 8) + 1

def fix_total_length(total_length, chunk_length, chunk_overlap):
    if total_length <= chunk_length: return force_wan_length(total_length)
    adjusted_chunk_length = chunk_length - chunk_overlap
    full_length_chunk_count = (total_length) // adjusted_chunk_length
    final_chunk_length = (total_length) % adjusted_chunk_length
    corrected_final_chunk_length = force_wan_length(final_chunk_length)
    return (full_length_chunk_count * adjusted_chunk_length) + corrected_final_chunk_length

def get_this_chunk_length(index, chunk_length, chunk_overlap, total_length):
    adjusted_chunk_length = chunk_length - chunk_overlap
    full_length_chunk_count = (total_length) // adjusted_chunk_length
    if index < full_length_chunk_count: return chunk_length
    return total_length - (adjusted_chunk_length * full_length_chunk_count)
