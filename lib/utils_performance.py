import time


def get_ts() -> int:
    return int(time.time() * 1000)  # current time in milliseconds
