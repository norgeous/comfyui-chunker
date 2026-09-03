from lib.calculate_progress_bar import calculate_progress_bar, _predict_delta


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


def test_cached_chunks_fallback_for_prediction():
    start = 1000
    chunk_lengths = [100, 45]
    result = calculate_progress_bar(
        execution_start_time=start + 4_000,
        chunk_start_times=[start],
        chunk_end_times=[start + 5_000],
        chunk_count=2,
        chunk_lengths=chunk_lengths,
    )
    assert result[0]["type"] == "cached"
    assert result[1]["type"] == "current"
    assert result[1]["delta"] == 1_000 / 100 * 45


def test_predict_delta_constant():
    assert _predict_delta([10, 10, 10], [1, 1, 1], 1) == 10


def test_predict_delta_linear_increase():
    assert _predict_delta([10, 20, 30], [1, 1, 1], 1) == 40


def test_predict_delta_linear_decrease():
    assert _predict_delta([40, 30, 20], [1, 1, 1], 1) == 10


def test_predict_delta_single_value():
    assert _predict_delta([10], [1], 1) == 10


def test_predict_delta_two_values():
    assert _predict_delta([10, 20], [1, 1], 1) == 30


def test_predict_delta_scales_by_chunk_length():
    assert _predict_delta([5_000, 5_000, 5_000], [100, 100, 100], 45) == 2_250


def test_predict_delta_empty():
    assert _predict_delta([], [], 1) is None


def test_predict_delta_multi_step():
    assert _predict_delta([10, 20, 30], [1, 1, 1], 1, 1) == 40
    assert _predict_delta([10, 20, 30], [1, 1, 1], 1, 2) == 50
    assert _predict_delta([10, 20, 30], [1, 1, 1], 1, 3) == 60


def test_multi_step_integration():
    start = 1000
    chunk_lengths = [100, 100, 100, 100, 100, 100]
    result = calculate_progress_bar(
        execution_start_time=start - 100_000,
        chunk_start_times=[start, start + 10_000, start + 20_000],
        chunk_end_times=[start + 3_000, start + 15_000, start + 27_000],
        chunk_count=6,
        chunk_lengths=chunk_lengths,
    )
    assert [r["type"] for r in result] == ["complete", "complete", "complete", "current", "pending", "pending"]
    assert [r["delta"] for r in result[:3]] == [3_000, 5_000, 7_000]
    assert result[3]["delta"] == 9_000
    assert result[4]["delta"] == 11_000
    assert result[5]["delta"] == 13_000
