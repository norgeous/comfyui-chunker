import execution

from .utils_performance import get_ts

_execution_start_time = 0


def get_execution_start_time():
    return _execution_start_time


def _patched_execute_async(*args, **kwargs):
    global _execution_start_time
    _execution_start_time = get_ts()
    return _original_execute_async(*args, **kwargs)


_original_execute_async = execution.PromptExecutor.execute_async
execution.PromptExecutor.execute_async = _patched_execute_async
