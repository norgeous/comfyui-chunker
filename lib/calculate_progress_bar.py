def calculate_progress_bar(create_time, chunk_start_times, chunk_end_times, chunk_count):
    delta_sum = 0
    delta_count = 0
    result = []
    for i in range(chunk_count):
        start_ts = chunk_start_times[i] if i < len(chunk_start_times) else None
        end_ts = chunk_end_times[i] if i < len(chunk_end_times) else None
        delta = end_ts - start_ts if start_ts is not None and end_ts is not None else None
        avg = delta_sum / delta_count if delta_count else None

        if start_ts is not None and start_ts < create_time:
            result.append({
                "type": "cached",
                "delta": end_ts - create_time,
            })
        elif delta is not None:
            delta_sum += delta
            delta_count += 1
            result.append({
                "type": "complete",
                "delta": delta,
            })
        else:
            result.append({
                "type": "current" if i == len(chunk_start_times) else "pending",
                "delta": avg,
            })
    return result
