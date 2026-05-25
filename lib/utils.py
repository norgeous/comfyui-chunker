from functools import reduce
from typing import Any, List


def log(*args: Any, **kwargs: Any) -> None:
    print(f"\U0001F36B  Chunker:", *args, **kwargs)


def count(items: List[Any]) -> int:
    if len(items) == 0:
        return 0
    if len(items) == 1:
        return len(items[0])
    return reduce(lambda acc, item: acc + len(item), [0, *items])
