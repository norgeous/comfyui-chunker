def calculate_progress_bar(execution_start_time, chunk_start_times, chunk_end_times, chunk_count, chunk_lengths):
    delta_sum = 0
    frame_sum = 0
    result = []
    for i in range(chunk_count):
        start_ts = chunk_start_times[i] if i < len(chunk_start_times) else None
        end_ts = chunk_end_times[i] if i < len(chunk_end_times) else None
        delta = end_ts - start_ts if start_ts is not None and end_ts is not None else None
        rate = delta_sum / frame_sum if frame_sum else None

        if start_ts is not None and start_ts < execution_start_time:
            delta = end_ts - execution_start_time
            delta_sum += delta
            frame_sum += chunk_lengths[i]
            result.append({
                "type": "cached",
                "delta": delta,
            })
        elif delta is not None:
            if result and result[-1]["type"] == "cached":
                delta_sum = 0
                frame_sum = 0
            delta_sum += delta
            frame_sum += chunk_lengths[i]
            result.append({
                "type": "complete",
                "delta": delta,
            })
        else:
            result.append({
                "type": "current" if i == len(chunk_start_times) else "pending",
                "delta": rate * chunk_lengths[i] if rate is not None else None,
            })
    return result
