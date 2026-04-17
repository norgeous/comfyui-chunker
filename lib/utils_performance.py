import time
import math

def get_ts():
    return int(time.time() * 1000) # current time in milliseconds

def predict(data, next_count):
    avg = sum(data) / len(data)
    return [avg] * next_count
