def calculate_progress_bar(execution_start_time, chunk_start_times, chunk_end_times, chunk_count, chunk_lengths, alpha=0.3):
    ema_rate = None
    result = []
    for i in range(chunk_count):
        start_ts = chunk_start_times[i] if i < len(chunk_start_times) else None
        end_ts = chunk_end_times[i] if i < len(chunk_end_times) else None
        delta = end_ts - start_ts if start_ts is not None and end_ts is not None else None

        if start_ts is not None and start_ts < execution_start_time:
            delta = end_ts - execution_start_time
            sample_rate = delta / chunk_lengths[i]
            ema_rate = sample_rate if ema_rate is None else alpha * sample_rate + (1 - alpha) * ema_rate
            result.append({
                "type": "cached",
                "delta": delta,
            })
        elif delta is not None:
            if result and result[-1]["type"] == "cached":
                ema_rate = None
            sample_rate = delta / chunk_lengths[i]
            ema_rate = sample_rate if ema_rate is None else alpha * sample_rate + (1 - alpha) * ema_rate
            result.append({
                "type": "complete",
                "delta": delta,
            })
        else:
            result.append({
                "type": "current" if i == len(chunk_start_times) else "pending",
                "delta": ema_rate * chunk_lengths[i] if ema_rate is not None else None,
            })
    return result
