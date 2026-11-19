"""Check benchmark statistics without making model-serving requests."""

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import statistics
import time

import pytest


@pytest.mark.parametrize(
    "latencies, median, p90, p99",
    [
        ([2.0], 2.0, 2.0, 2.0),
        ([3.0, 1.0], 2.0, 2.8, 2.98),
        (list(range(1, 21)), 10.5, 18.1, 19.81),
    ],
)
def test_benchmark_reports_interpolated_latency_percentiles(latencies, median, p90, p99):
    notebook = Path(__file__).resolve().parents[1] / "src/models/medgemma/xgrammar/03_benchmark_endpoint.py"
    definition = next(
        node for node in ast.parse(notebook.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "run_level"
    )
    namespace = {
        "WARMUP_REQUESTS": 0,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "as_completed": as_completed,
        "send_request": lambda latency: (latency, 1, 10, 20, None),
        "statistics": statistics,
        "time": time,
    }
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(notebook), "exec"), namespace)
    metrics = namespace["run_level"](2, latencies)

    assert metrics["latency_median_s"] == pytest.approx(median)
    assert metrics["latency_p90_s"] == pytest.approx(p90)
    assert metrics["latency_p99_s"] == pytest.approx(p99)
    assert metrics["latency_min_s"] <= median <= p90 <= p99 <= metrics["latency_max_s"]
    assert metrics["total_requests"] == len(latencies)
    assert metrics["error_count"] == 0
