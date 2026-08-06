import functools
import gc
import json
import logging
import math
import os
import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Union
import dataclasses
import ray
from ray._private.internal_api import get_memory_info_reply, get_state_from_address
from ray.util.state import list_runtime_envs

logger = logging.getLogger(__name__)


def _get_spilled_bytes_total(state) -> float:
    """Get the total number of spilled bytes across the cluster."""
    return get_memory_info_reply(state).store_stats.spilled_bytes_total


def _bytes_to_gb(b: float) -> float:
    return round(b / (1024**3), 4)


class ObjectStoreMemorySampler:
    """Samples aggregate object store usage and tracks the peak value.

    Object store usage is an instantaneous gauge, so checking only at the
    beginning and end of a benchmark can miss short-lived memory spikes.
    """

    def __init__(self, state, interval_s: float = 1.0):
        self._state = state
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread = None

        self._peak_used_bytes = 0
        self._peak_utilization = 0.0

    @property
    def peak_used_bytes(self) -> int:
        return self._peak_used_bytes

    @property
    def peak_utilization(self) -> float:
        return self._peak_utilization

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def start(self):
        self._sample_once()
        self._thread = threading.Thread(
            target=self._run,
            name="object-store-memory-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._sample_once()

    def _run(self):
        while not self._stop_event.wait(self._interval_s):
            self._sample_once()

    def _sample_once(self):
        try:
            store_stats = get_memory_info_reply(self._state).store_stats
        except Exception:
            logger.warning("Failed to sample object store memory.", exc_info=True)
            return

        used_bytes = store_stats.object_store_bytes_used
        capacity_bytes = store_stats.object_store_bytes_avail

        self._peak_used_bytes = max(self._peak_used_bytes, used_bytes)

        if capacity_bytes > 0:
            self._peak_utilization = max(
                self._peak_utilization,
                used_bytes / capacity_bytes,
            )


def _pin_ray_address() -> None:
    """Point the State API at *our* Ray instance.

    Single-machine runs use ``RAY_ADDRESS=local`` so ``ray.init()`` can never
    attach to an Anyscale workspace's managed cluster. But the State API
    re-resolves the address from the environment, and "local" is ambiguous when
    two instances are up -- ours and the workspace's -- so it raises
    ``ConnectionError: Found multiple active Ray instances``. The driver knows
    its own GCS address; pinning it makes the resolution unambiguous. No-op on a
    real cluster, where there is only one instance and RAY_ADDRESS is already set.
    """
    try:
        gcs = ray.get_runtime_context().gcs_address
    except Exception:
        return
    if gcs:
        os.environ["RAY_ADDRESS"] = gcs


def _stats_summary(ds: "ray.data.Dataset", detail: bool = True):
    """``get_stats_summary``, degrading instead of failing.

    ``detail=True`` additionally queries the State API for per-operator
    scheduling overhead. That query can fail for reasons unrelated to anything
    we measure -- no dashboard/API server, task events disabled, address
    ambiguity -- and it would take the per-task memory distributions down with
    it, which are the entire point of these runs. So fall back to the summary
    without scheduling overhead rather than losing every metric.
    """
    _pin_ray_address()
    if detail:
        try:
            return ds.get_stats_summary(detail=True)
        except Exception:
            logger.warning(
                "get_stats_summary(detail=True) failed; retrying without "
                "scheduling overhead (memory metrics are unaffected)",
                exc_info=True,
            )
    return ds.get_stats_summary(detail=False)


def collect_dataset_stats(ds: "ray.data.Dataset") -> Dict[str, Any]:
    """Collect execution stats from a Dataset as a JSON-serializable dict.
    This is a subset from `get_stats_summary`, because we are only adding the ones
    we care about for the release tests."""
    summary = _stats_summary(ds)
    return {
        "total_scheduling_runtime": summary.streaming_exec_schedule_s,
        "avg_scheduling_loop_duration_s": summary.streaming_exec_schedule_avg_s,
        "max_scheduling_loop_duration_s": summary.streaming_exec_schedule_max_s,
        "p50_scheduling_loop_duration_s": summary.streaming_exec_schedule_p50_s,
        "p90_scheduling_loop_duration_s": summary.streaming_exec_schedule_p90_s,
        "operators": [
            {
                "operator_name": op.operator_name,
                "earliest_start_time": op.earliest_start_time,
                "latest_end_time": op.latest_end_time,
                "scheduling_overhead": (
                    [dataclasses.asdict(bucket) for bucket in op.scheduling_overhead]
                    if op.scheduling_overhead
                    else []
                ),
            }
            for op in summary.operators_stats
        ],
    }


def collect_operator_metrics(ds: "ray.data.Dataset") -> Dict[str, Any]:
    """Per-operator time / output-bytes / worker-memory, for merging into a result dict.

    Surfaces numbers that otherwise live only on the Prometheus dashboard, not in the
    release log or databricks: each operator's wall time, output size/rows, and its
    per-task peak worker memory — USS (private working set) and RSS (OS-visible
    footprint, includes mapped object-store pages), both as the average across tasks
    and the single worst task. All four come from ``MemoryProfiler`` sampling inside
    the task (Linux-only; ``None`` on macOS). This isolates the read operator's cost
    from downstream compute and exposes the decode-memory metrics that the aggregate
    object-store peak cannot see. Best-effort: returns a partial/empty dict rather
    than failing the benchmark.

    A ``read_*`` top-level convenience is filled from the first ``Read*`` operator so
    the "parquet part" (read wall time + output bytes + decode USS/RSS) is a
    first-class field.

    NOTE: stats attach to the consumed dataset handle. Consume ``ds`` itself
    (``iter_*``/``write_*``/``materialize``) before calling this.

    ``ds.count()`` reports nothing, for two stacked reasons: on a plain
    ``read_parquet`` dataset ``Dataset.count`` returns ``_meta_count()`` straight
    from the footer row counts (``dataset.py:4114``) and **executes nothing at
    all** — no decode, no tasks, no stats; and even when it does fall through, it
    builds a ``Count`` op over a *copy* of the plan, leaving ``ds`` unexecuted.
    So a ``--count`` benchmark measures footer reading, not decoding. Use
    ``--iter-batches`` (decodes everything, retains ~one batch) when you want the
    decode working set, or ``--iter-bundles`` when you want retention too.
    """
    from ray.data._internal.stats import DatasetStatsSummary

    def _sum(stat) -> Any:
        return stat.sum if stat is not None else None

    def _spread(stat, prefix: str) -> Dict[str, Any]:
        """Flatten a ``StatsSummary`` into ``<prefix>_{count,min,max,mean}`` keys.

        Ray already accumulates these; nothing here computes anything new. See
        ``OperatorStatsSummary.task_rows`` / ``.node_count``.
        """
        if stat is None:
            return {f"{prefix}_{k}": None for k in ("count", "min", "max", "mean")}
        return {
            f"{prefix}_count": stat.count,
            f"{prefix}_min": stat.min,
            f"{prefix}_max": stat.max,
            f"{prefix}_mean": stat.mean,
        }

    # (result-dict key, extra_metrics key) for the per-task memory metrics.
    mem_keys = [
        ("avg_max_uss_per_task_bytes", "average_max_uss_per_task"),
        ("max_uss_per_task_bytes", "max_uss_per_task"),
        ("avg_max_rss_per_task_bytes", "average_max_rss_per_task"),
        ("max_rss_per_task_bytes", "max_rss_per_task"),
    ]
    # Distributions behind those scalars. ``average_max_uss_per_task`` returns
    # None at zero samples, which renders identically to "memory was flat" — so
    # always report the sample count and the spread alongside it. A row with
    # ``uss_num_samples: 0`` is an instrumentation gap; a row with 1 is a single
    # task, where avg == max and no spread is knowable. Neither is comparable to
    # a many-sample row, and without this field you cannot tell them apart.
    dist_keys = [("uss", "max_uss_bytes"), ("rss", "max_rss_bytes")]
    dist_stats = ("num_samples", "mean", "min", "max", "p50", "p90", "p99")
    # Promoted to top-level ``read_*`` alongside the memory metrics.
    decomposition_keys = [
        f"{p}_{k}"
        for p in ("task_rows", "node_count")
        for k in ("count", "min", "max", "mean")
    ] + ["earliest_start_time", "latest_end_time", "time_total_s"]

    out: Dict[str, Any] = {"operators_detail": []}
    try:
        summary = _stats_summary(ds)
        for node in DatasetStatsSummary._collect_dataset_stats_summaries(summary):
            extra = getattr(node, "extra_metrics", {}) or {}
            mem = {out_key: extra.get(in_key) for out_key, in_key in mem_keys}
            for prefix, dist_key in dist_keys:
                dist = extra.get(dist_key)
                dist = dist if isinstance(dist, dict) else {}
                # num_samples defaults to 0 (not None): "no samples" is a fact we
                # know, unlike a percentile that needs the datasketches extra.
                mem[f"{prefix}_num_samples"] = dist.get("num_samples", 0)
                for stat in dist_stats[1:]:
                    mem[f"{prefix}_{stat}_bytes"] = dist.get(stat)
            for op in node.operators_stats or []:
                # How the work was decomposed and where it ran. Ray accumulates
                # both already (``rows_per_task`` keyed by task index,
                # ``tasks_per_node`` keyed by node id) but nothing surfaced them,
                # so every comparison so far has had to *infer* task count from
                # ``uss_num_samples`` and could say nothing at all about
                # placement. ``task_rows_count`` is the read-task count;
                # ``task_rows_{min,max}`` is how evenly the input was split;
                # ``node_count_count`` is how many nodes participated and
                # ``node_count_{min,max}`` how lopsided the spread was.
                #
                # This is what makes a single-node run comparable to a 10-node
                # one: without it, a ratio difference between the two cannot be
                # attributed to the reader rather than to the decomposition.
                decomposition = {
                    **_spread(op.task_rows, "task_rows"),
                    **_spread(op.node_count, "node_count"),
                    "earliest_start_time": op.earliest_start_time,
                    "latest_end_time": op.latest_end_time,
                    "time_total_s": op.time_total_s,
                }
                out["operators_detail"].append(
                    {
                        "operator_name": op.operator_name,
                        "wall_time_s": _sum(op.wall_time),
                        "cpu_time_s": _sum(op.cpu_time),
                        "udf_time_s": _sum(op.udf_time),
                        "output_num_rows": _sum(op.output_num_rows),
                        "output_size_bytes": _sum(op.output_size_bytes),
                        **decomposition,
                        **mem,
                    }
                )
        for entry in out["operators_detail"]:
            if "Read" in (entry["operator_name"] or ""):
                out["read_operator_name"] = entry["operator_name"]
                out["read_wall_time_s"] = entry["wall_time_s"]
                out["read_output_size_bytes"] = entry["output_size_bytes"]
                out["read_output_num_rows"] = entry["output_num_rows"]
                # Decoded bytes per row: which side of the arrow-rs decode-budget
                # floor this shape falls on (the byte budget stops binding above
                # ~budget/2048 bytes per row), so a regression can be attributed
                # to row width without re-reading the fixture's footer.
                rows, nbytes = entry["output_num_rows"], entry["output_size_bytes"]
                out["read_decoded_bytes_per_row"] = (
                    (nbytes / rows) if (rows and nbytes) else None
                )
                for out_key, _ in mem_keys:
                    out[f"read_{out_key}"] = entry[out_key]
                for prefix, _ in dist_keys:
                    out[f"read_{prefix}_num_samples"] = entry[f"{prefix}_num_samples"]
                    for stat in dist_stats[1:]:
                        out[f"read_{prefix}_{stat}_bytes"] = entry[
                            f"{prefix}_{stat}_bytes"
                        ]
                for key in decomposition_keys:
                    out[f"read_{key}"] = entry.get(key)
                break
    except Exception:
        logger.warning("collect_operator_metrics failed", exc_info=True)
    return out


def with_operator_metrics(benchmark_fn):
    """Wrap a ``benchmark_fn`` so :func:`collect_operator_metrics` runs against the
    last dataset the function materialized, and merge the result into its dict.

    Stats live on the executed handle, which most drivers never return — they end
    in a bare ``....materialize()`` inside an expression statement and then
    ``return vars(args)``. Threading the handle out of each of them would be ~20
    near-identical edits; capturing it here is one. During the call
    ``Dataset.materialize`` is temporarily wrapped to remember what it returned,
    and restored in a ``finally`` so a failing benchmark cannot leave the patch
    installed.

    No-ops (returns the fn's dict unchanged) if the fn materialized nothing or
    returned a non-dict, so it is safe to apply to any driver.
    """

    @functools.wraps(benchmark_fn)
    def wrapper(*args, **kwargs):
        from ray.data import Dataset

        captured = {}
        original = Dataset.materialize

        def spy(self, *a, **kw):
            out = original(self, *a, **kw)
            captured["ds"] = out
            return out

        Dataset.materialize = spy
        try:
            result = benchmark_fn(*args, **kwargs)
        finally:
            Dataset.materialize = original

        if isinstance(result, dict) and "ds" in captured:
            return {**result, **collect_operator_metrics(captured["ds"])}
        return result

    return wrapper


class RuntimeEnvSetupTracker:
    """Collects runtime environment creation times across the cluster.

    Queries the Ray State API for all runtime environments and reports
    aggregate statistics (mean, stdev) for creation time.

    Usage::

        # After a pipeline or job completes:
        stats = RuntimeEnvSetupTracker.collect()
    """

    @staticmethod
    def collect() -> List[Dict[str, Any]]:
        try:
            groups: Dict[str, List[float]] = {}
            for env in list_runtime_envs(limit=1000):
                if env.creation_time_ms is None:
                    continue
                label = "+".join(sorted(env.runtime_env.keys()))
                groups.setdefault(label, []).append(env.creation_time_ms)
        except Exception:
            logger.warning("Failed to query runtime env creation times.", exc_info=True)
            return []

        results: List[Dict[str, Any]] = []
        for label, times in groups.items():
            mean = sum(times) / len(times)
            variance = sum((t - mean) ** 2 for t in times) / len(times)
            results.append(
                {
                    "runtime_env_type": label,
                    "count": len(times),
                    "mean_creation_time_ms": round(mean, 2),
                    "stdev_creation_time_ms": round(math.sqrt(variance), 2),
                }
            )
        return results


def benchmark_py_modules() -> List[str]:
    """Return paths to benchmark.py and the profiling
    package for use in runtime_env py_modules."""
    dataset_dir = os.path.dirname(os.path.realpath(__file__))
    return [
        os.path.realpath(__file__),
        os.path.join(dataset_dir, "profiling"),
    ]


class BenchmarkMetric(Enum):
    RUNTIME = "time"
    NUM_ROWS = "num_rows"
    THROUGHPUT = "tput"
    ACCURACY = "accuracy"
    OBJECT_STORE_SPILLED_TOTAL_GB = "object_store_spilled_total_gb"
    OBJECT_STORE_MEMORY_USED_PEAK_GB = "object_store_memory_used_peak_gb"
    OBJECT_STORE_MEMORY_UTILIZATION_PEAK = "object_store_memory_utilization_peak"


class Benchmark:
    """Runs benchmarks in a way that's compatible with our release test infrastructure.

    Here's an example of typical usage:

    .. testcode::

        import time
        from benchmark import Benchmark

        def sleep(sleep_s)
            time.sleep(sleep_s)
            # Return any extra metrics you want to record. This can include
            # configuration parameters, accuracy, etc.
            return {"sleep_s": sleep_s}

        benchmark = Benchmark()
        benchmark.run_fn("short", sleep, 1)
        benchmark.run_fn("long", sleep, 10)
        benchmark.write_result()

    This code outputs a JSON file with contents like this:

    .. code-block:: json

        {"short": {"time": 1.0, "sleep_s": 1}, "long": {"time": 10.0 "sleep_s": 10}}
    """

    def __init__(self):
        self.result = {}

    def run_fn(
        self,
        name: str,
        fn: Callable[..., Dict[Union[str, BenchmarkMetric], Any]],
        *fn_args,
        **fn_kwargs,
    ):
        """Benchmark a function.

        This is the most general benchmark utility available. Use it if the other
        methods are too specific.

        ``run_fn`` automatically records the runtime of ``fn``. To report additional
        metrics, return a ``Dict[str, Any]`` of metric labels to metric values from your
        function.
        """
        gc.collect()

        print(f"Running case: {name}")
        state = get_state_from_address(ray.get_runtime_context().gcs_address)

        with ObjectStoreMemorySampler(state) as memory_sampler:
            start_time = time.perf_counter()
            start_spilled_bytes = _get_spilled_bytes_total(state)

            try:
                fn_output = fn(*fn_args, **fn_kwargs)
            finally:
                duration = time.perf_counter() - start_time

        assert fn_output is None or isinstance(fn_output, dict), fn_output

        spilled_bytes_total = _get_spilled_bytes_total(state) - start_spilled_bytes
        curr_case_metrics = {
            BenchmarkMetric.RUNTIME.value: duration,
            BenchmarkMetric.OBJECT_STORE_SPILLED_TOTAL_GB.value: _bytes_to_gb(
                spilled_bytes_total
            ),
            BenchmarkMetric.OBJECT_STORE_MEMORY_USED_PEAK_GB.value: _bytes_to_gb(
                memory_sampler.peak_used_bytes
            ),
            BenchmarkMetric.OBJECT_STORE_MEMORY_UTILIZATION_PEAK.value: round(
                memory_sampler.peak_utilization,
                4,
            ),
        }
        if isinstance(fn_output, dict):
            for key, value in fn_output.items():
                if isinstance(key, BenchmarkMetric):
                    curr_case_metrics[key.value] = value
                elif isinstance(key, str):
                    curr_case_metrics[key] = value
                else:
                    raise ValueError(f"Unexpected metric key type: {type(key)}")

        self.result[name] = curr_case_metrics
        print(f"Result of case {name}: {curr_case_metrics}")

    def write_result(self):
        """Write all results to the appropriate JSON file.

        Our release test infrastructure consumes the JSON file and uploads the results
        to our internal dashboard.
        """
        # 'TEST_OUTPUT_JSON' is set in the release test environment.
        test_output_json = os.environ.get("TEST_OUTPUT_JSON", "./result.json")
        with open(test_output_json, "w") as f:
            f.write(json.dumps(self.result))

        print(f"Finished benchmark, metrics exported to '{test_output_json}':")
        print(json.dumps(self.result, indent=4))
