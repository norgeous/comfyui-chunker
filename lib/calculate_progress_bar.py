def _predict_delta(deltas, source_chunk_lengths, target_chunk_length, steps_ahead=1):
    if not deltas:
        return None
    rates = [d / l for d, l in zip(deltas, source_chunk_lengths)]
    for _ in range(steps_ahead):
        if len(rates) == 1:
            rates.append(rates[-1])
        elif len(rates) == 2:
            rates.append(2 * rates[-1] - rates[-2])
        else:
            r1, r2, r3 = rates[-3], rates[-2], rates[-1]
            rates.append(3 * (r3 - r2) + r1)
    return rates[-1] * target_chunk_length


def calculate_progress_bar(execution_start_time, chunk_start_times, chunk_end_times, chunk_count, chunk_lengths):
    completed_deltas = []
    completed_lengths = []
    pending_count = 0
    result = []
    for i in range(chunk_count):
        start_ts = chunk_start_times[i] if i < len(chunk_start_times) else None
        end_ts = chunk_end_times[i] if i < len(chunk_end_times) else None
        delta = end_ts - start_ts if start_ts is not None and end_ts is not None else None

        if start_ts is not None and start_ts < execution_start_time:
            delta = end_ts - execution_start_time
            result.append({
                "type": "cached",
                "delta": delta,
            })
        elif delta is not None:
            completed_deltas.append(delta)
            completed_lengths.append(chunk_lengths[i])
            result.append({
                "type": "complete",
                "delta": delta,
            })
        else:
            predicted = _predict_delta(completed_deltas, completed_lengths, chunk_lengths[i], pending_count + 1)
            pending_count += 1
            result.append({
                "type": "current" if i == len(chunk_start_times) else "pending",
                "delta": predicted,
            })
    return result
