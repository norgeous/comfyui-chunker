from functools import reduce

def log(*args, **kwargs):
    print(f"\U0001F36B  Chunker:", *args, **kwargs)

def count(list):
    if len(list) == 0: return 0
    if len(list) == 1: return len(list[0])
    return reduce(lambda acc, item: acc + len(item), [0, *list])
