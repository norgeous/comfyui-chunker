from lib.calculate_progress_bar import calculate_progress_bar


def test_equal_lengths_reproduces_avg():
    start = 1000
    chunk_lengths = [100, 100, 100, 100]
    result = calculate_progress_bar(
        execution_start_time=start - 100_000,
        chunk_start_times=[start, start + 10_000, start + 20_000],
        chunk_end_times=[start + 5_000, start + 15_000, start + 25_000],
        chunk_count=4,
        chunk_lengths=chunk_lengths,
    )
    assert [r["type"] for r in result] == ["complete", "complete", "complete", "current"]
    assert [r["delta"] for r in result[:3]] == [5_000, 5_000, 5_000]
    assert result[3]["delta"] == 5_000


def test_variable_lengths_scale_current_eta():
    start = 1000
    chunk_lengths = [100, 100, 100, 45]
    result = calculate_progress_bar(
        execution_start_time=start - 100_000,
        chunk_start_times=[start, start + 10_000, start + 20_000],
        chunk_end_times=[start + 5_000, start + 15_000, start + 25_000],
        chunk_count=4,
        chunk_lengths=chunk_lengths,
    )
    assert result[3]["type"] == "current"
    assert result[3]["delta"] == 5_000 / 100 * 45


def test_pending_chunks_scale_by_length():
    start = 1000
    chunk_lengths = [100, 100, 100, 100, 45]
    result = calculate_progress_bar(
        execution_start_time=start - 100_000,
        chunk_start_times=[start],
        chunk_end_times=[start + 5_000],
        chunk_count=5,
        chunk_lengths=chunk_lengths,
    )
    assert [r["type"] for r in result] == ["complete", "current", "pending", "pending", "pending"]
    assert result[2]["delta"] == 5_000 / 100 * 100
    assert result[4]["delta"] == 5_000 / 100 * 45


def test_cached_chunks_reset_rate():
    start = 1000
    chunk_lengths = [100, 100, 45]
    result = calculate_progress_bar(
        execution_start_time=start + 4_000,
        chunk_start_times=[start, start + 10_000],
        chunk_end_times=[start + 5_000, start + 15_000],
        chunk_count=3,
        chunk_lengths=chunk_lengths,
    )
    assert result[0]["type"] == "cached"
    assert result[1]["type"] == "complete"
    assert result[2]["type"] == "current"
    assert result[2]["delta"] == 5_000 / 100 * 45


def test_ema_weights_recent_chunk_higher():
    start = 1000
    chunk_lengths = [100, 100, 100]
    result = calculate_progress_bar(
        execution_start_time=start - 100_000,
        chunk_start_times=[start, start + 20_000],
        chunk_end_times=[start + 10_000, start + 25_000],
        chunk_count=3,
        chunk_lengths=chunk_lengths,
        alpha=0.3,
    )
    slow_rate = 10_000 / 100  # chunk 0: 100ms/frame
    fast_rate = 5_000 / 100   # chunk 1: 50ms/frame
    ema_rate = 0.3 * fast_rate + 0.7 * slow_rate
    assert result[2]["type"] == "current"
    assert result[2]["delta"] == ema_rate * 100
    assert slow_rate > ema_rate > fast_rate
