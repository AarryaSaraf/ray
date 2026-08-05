"""Single-machine A/B + knob-sweep harness for the arrow-rs Parquet reader.

Runs one (reader x fixture x knobs x consumer) configuration per subprocess, so
each gets a clean ``DataContext`` and a clean process-memory baseline, and reports
wall time, per-task decode memory, cluster-wide peak RSS, and the reader's own
profiling records side by side.

Why a separate harness rather than the release suite: the regressions we are
chasing live in the crate's **S3** planner (``plan_s3_units`` / the prefetch
admission loop), which the local ``read_row_groups`` entry point never touches --
so a local-filesystem benchmark provably cannot reproduce them. Pointing this at
an S3-compatible endpoint (MinIO or moto) on the same box exercises the real S3
code path with no cloud account and no cluster. See ``--endpoint``.

Examples::

    # H3 (batch floor) -- local files, no S3 needed
    python arrow_rs_probe.py --data /data/fix --shapes fat_rows,narrow_big_rg \\
        --readers arrow_rs,pyarrow

    # H2 (oversized unit serializes fetch) -- needs the S3 path
    python arrow_rs_probe.py --data s3://bench/fix --endpoint http://127.0.0.1:9000 \\
        --shapes fat_col,fat_col_solo --readers arrow_rs,pyarrow

    # H2 discriminator: same bytes, one column vs two
    python arrow_rs_probe.py ... --shapes fat_col --columns payload
    python arrow_rs_probe.py ... --shapes fat_col --columns payload,label

    # knob sweep (H3/H4/H6/H7)
    python arrow_rs_probe.py ... --readers arrow_rs \\
        --sweep arrow_rs_decode_budget_bytes=2097152,8388608,33554432,134217728

Output: one JSON object per run on stdout (``kind: "run"``), then a summary table
on stderr. Redirect stdout to a file to keep the raw records.
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

MiB = 1024 * 1024

# Knobs a sweep may set. Passed through ``dataset_kwargs``, where the reader
# resolves them as kwarg > env > default (``_ArrowRsTuning``).
KNOBS = (
    "arrow_rs_decode_budget_bytes",
    "arrow_rs_k",
    "arrow_rs_split_threshold_bytes",
    "arrow_rs_fetch_window_mb",
    "arrow_rs_column_fetch_mb",
    "arrow_rs_prefetch_budget_mb",
)


# --------------------------------------------------------------------------- #
# Cluster-wide RSS sampler
# --------------------------------------------------------------------------- #
class WorkerRssSampler:
    """Samples summed RSS across Ray worker processes and keeps the peak.

    Per-task USS (``MemoryProfiler``) answers "how much did one decode hold";
    this answers "how close did the machine come to OOM", which is the claim the
    migration is actually about. They differ: USS is sampled inside a task and
    misses concurrency, while this sums every worker at an instant.

    RSS double-counts pages shared between workers (the interpreter, PyArrow's
    and the crate's ``.so``), so treat the sum as an upper bound and compare
    arms, never absolutes.
    """

    def __init__(self, interval_s: float = 0.05):
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_sum_bytes = 0
        self.peak_single_bytes = 0
        self.peak_driver_bytes = 0
        self.peak_num_workers = 0
        self.trace: List[Tuple[float, int, int]] = []  # (t, sum_rss, n_workers)
        self._t0 = time.perf_counter()

    def __enter__(self):
        try:
            import psutil  # noqa: F401
        except ImportError:
            print(
                "arrow_rs_probe: psutil missing, skipping RSS sampling", file=sys.stderr
            )
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # The raylet's own cmdline embeds the worker launch template, so matching
    # "default_worker.py" alone picks up the raylet (a ~27 MiB supervisor) and
    # misses the actual workers -- which reports a decode of 1 GiB as using
    # 27 MiB. Match the interpreter processes only, and never these.
    _NOT_WORKERS = ("raylet", "gcs_server", "plasma_store", "dashboard")

    # Ray rewrites a worker's argv[0] to "ray::IDLE" (or "ray::<Task>" while
    # running), and psutil's `name` stays the interpreter ("python3.12") -- so the
    # marker is in the *cmdline*, not the name. These are Ray's own infra actors,
    # not data workers: they add a constant few hundred MiB to both arms.
    _INFRA_ACTORS = (
        "_StatsActor",
        "_AutoscalingCoordinatorActor",
        "DashboardAgent",
        "JobSupervisor",
    )

    def _is_worker(self, proc) -> bool:
        name = (proc.info.get("name") or "").lower()
        if any(skip in name for skip in self._NOT_WORKERS):
            return False
        cmdline = proc.info.get("cmdline") or ()
        if not cmdline:
            return False
        marker = cmdline[0]
        if marker.startswith("ray::"):
            return not any(actor in marker for actor in self._INFRA_ACTORS)
        # Workers whose argv was not rewritten: argv[0] must be the interpreter
        # running the worker entrypoint, so the raylet (which merely *mentions*
        # the entrypoint in its launch template) does not match.
        return "python" in os.path.basename(marker).lower() and any(
            "default_worker.py" in arg for arg in cmdline
        )

    def _run(self):
        import psutil

        me = os.getpid()
        while not self._stop.wait(self._interval_s):
            total = biggest = count = 0
            for proc in psutil.process_iter(["name", "cmdline", "pid"]):
                try:
                    if proc.info["pid"] == me or not self._is_worker(proc):
                        continue
                    rss = proc.memory_info().rss
                except Exception:  # process died mid-iteration, or no permission
                    continue
                total += rss
                biggest = max(biggest, rss)
                count += 1
            try:
                driver_rss = psutil.Process(me).memory_info().rss
            except Exception:
                driver_rss = 0
            self.peak_driver_bytes = max(self.peak_driver_bytes, driver_rss)
            if count:
                self.peak_sum_bytes = max(self.peak_sum_bytes, total)
                self.peak_single_bytes = max(self.peak_single_bytes, biggest)
                self.peak_num_workers = max(self.peak_num_workers, count)
            self.trace.append((round(time.perf_counter() - self._t0, 3), total, count))

    def result(self) -> Dict[str, Any]:
        return {
            "worker_peak_sum_rss_bytes": self.peak_sum_bytes or None,
            "worker_peak_single_rss_bytes": self.peak_single_bytes or None,
            "worker_peak_count": self.peak_num_workers or None,
            # The consumer lives here, so the driver is part of the machine's
            # high-water mark even though it decodes nothing.
            "driver_peak_rss_bytes": self.peak_driver_bytes or None,
            "worker_rss_samples": len(self.trace),
        }


# --------------------------------------------------------------------------- #
# Profiling-record aggregation
# --------------------------------------------------------------------------- #
def summarize_profile(prof_dir: str) -> Dict[str, Any]:
    """Fold the JSONL records the reader and the crate wrote into per-hypothesis
    scalars. This is the part that turns "it was slow" into "it took the
    column-group path with N oversized units and retained M bytes"."""
    records: List[dict] = []
    if os.path.isdir(prof_dir):
        for name in sorted(os.listdir(prof_dir)):
            if not name.endswith(".jsonl"):
                continue
            with open(os.path.join(prof_dir, name)) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    by_kind: Dict[str, List[dict]] = {}
    for rec in records:
        by_kind.setdefault(rec.get("kind", "?"), []).append(rec)

    plans, rgs = by_kind.get("s3_plan", []), by_kind.get("s3_rg", [])
    local, frags = by_kind.get("local_rg", []), by_kind.get("fragment", [])
    fallbacks = by_kind.get("fallback", [])

    def total(recs, key):
        vals = [r.get(key) for r in recs if isinstance(r.get(key), (int, float))]
        return sum(vals) if vals else None

    def biggest(recs, key):
        vals = [r.get(key) for r in recs if isinstance(r.get(key), (int, float))]
        return max(vals) if vals else None

    hstack = [r for r in rgs if r.get("mode") == "column_groups"]
    out: Dict[str, Any] = {
        "prof_num_records": len(records),
        "prof_fragments": len(frags),
        "prof_fallbacks": len(fallbacks),
        "prof_fallback_reasons": sorted({r.get("reason", "?") for r in fallbacks}),
        # --- H2: did any fetch unit exceed the whole prefetch budget? ---
        "prof_s3_row_group_plans": len(plans),
        "prof_oversized_units": total(plans, "oversized_units"),
        "prof_max_unit_kib": biggest(plans, "unit_kib_max"),
        "prof_col_group_rgs": total(plans, "col_group_rgs"),
        "prof_row_window_rgs": total(plans, "row_window_rgs"),
        # --- H1: how many decoded bytes were held at once to hstack? ---
        "prof_hstack_row_groups": len(hstack),
        "prof_max_retained_bytes": biggest(rgs, "retained_bytes"),
        "prof_max_rg_uncompressed_bytes": biggest(rgs + local, "rg_uncompressed_bytes"),
        # --- fetch vs decode split (is it latency-bound or CPU-bound?) ---
        "prof_fetch_wait_s": total(rgs, "fetch_wait_s"),
        "prof_decode_s": total(rgs, "decode_s"),
        # --- H3: did the 2048-row floor override the byte budget? ---
        "prof_floored_row_groups": sum(1 for r in local if r.get("floored")),
        "prof_local_row_groups": len(local),
        "prof_max_eff_batch_bytes": biggest(local, "eff_batch_bytes"),
        "prof_max_peak_batch_bytes": biggest(frags, "peak_batch_bytes"),
    }
    # The single most diagnostic ratio: retained decoded bytes vs the row group.
    retained, rg_size = out["prof_max_retained_bytes"], biggest(
        rgs, "rg_uncompressed_bytes"
    )
    if retained and rg_size:
        out["prof_retained_over_row_group"] = round(retained / rg_size, 3)
    return out


# --------------------------------------------------------------------------- #
# One run (child process)
# --------------------------------------------------------------------------- #
def _prepare_env() -> None:
    """Neutralise the managed-workspace traps before importing/starting Ray.

    Each of these cost real debugging time on the 2026-07 boxes (runbook §2):

    * ``RAY_RUNTIME_ENV_HOOK`` / ``RAY_RUNTIME_ENV_PLUGINS`` -- Anyscale images
      inject a cgroup module that is not in this venv. With the *plugins* variant
      the runtime-env agent dies on import, the raylet fate-shares, and
      ``ray.init()`` hangs forever with no error on stdout.
    * ``RAY_ADDRESS=local`` -- otherwise ``ray.init()`` attaches to the managed
      cluster, which runs a different Ray version and a different reader.
    * ``RAY_task_events_report_interval_ms=0`` -- some 2026-07 master nightlies
      SIGSEGV in the core task-event aggregator flush, killing workers *and* the
      driver. The harness does not read task events.
    """
    os.environ.pop("RAY_RUNTIME_ENV_HOOK", None)
    os.environ.pop("RAY_RUNTIME_ENV_PLUGINS", None)
    os.environ.setdefault("RAY_ADDRESS", "local")
    os.environ.setdefault("RAY_task_events_report_interval_ms", "0")


def _assert_local_ray() -> None:
    """Refuse to measure a Ray that is not this checkout.

    In a fresh shell without the venv activated, ``python`` is the image's
    anaconda, whose Ray is the Anyscale runtime -- a different read path
    entirely, where this branch's reader and flags do not exist. The run would
    complete and the numbers would be meaningless.
    """
    if os.environ.get("RAY_DATA_BENCH_ALLOW_FOREIGN_RAY") == "1":
        return
    import ray.data

    resolved = os.path.realpath(ray.data.__file__)
    repo = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if not resolved.startswith(repo + os.sep):
        raise SystemExit(
            f"ray.data resolves to {resolved}, which is outside this checkout "
            f"({repo}). Activate the venv (source <repo>/.venv/bin/activate) so "
            "setup-dev.py's symlink is in effect, or set "
            "RAY_DATA_BENCH_ALLOW_FOREIGN_RAY=1 if that is deliberate."
        )


def run_one(args: argparse.Namespace) -> Dict[str, Any]:
    _prepare_env()
    import ray
    from benchmark import collect_operator_metrics

    _assert_local_ray()

    prof_dir = args.prof_dir
    if prof_dir:
        os.makedirs(prof_dir, exist_ok=True)
        os.environ["RAY_DATA_ARROW_RS_PROFILE"] = "1"
        os.environ["RAY_DATA_ARROW_RS_PROFILE_DIR"] = prof_dir

    # Pin the cluster shape: an A/B is only comparable if both arms get the same
    # CPUs and the same object-store budget, and the defaults derive from free
    # RAM, which drifts between runs on the same box.
    init_kwargs: Dict[str, Any] = {
        "address": "local",
        "include_dashboard": False,
        "log_to_driver": False,
        "ignore_reinit_error": True,
    }
    if args.num_cpus:
        init_kwargs["num_cpus"] = args.num_cpus
    if args.object_store_mb:
        init_kwargs["object_store_memory"] = args.object_store_mb * MiB
    ray.init(**init_kwargs)
    from ray.data import DataContext

    ctx = DataContext.get_current()
    ctx.use_datasource_v2 = True
    ctx.use_arrow_rs_parquet_reader = args.reader == "arrow_rs"
    if args.target_block_size_mib:
        ctx.target_max_block_size = args.target_block_size_mib * MiB

    dataset_kwargs: Dict[str, Any] = {}
    for knob in KNOBS:
        value = getattr(args, knob, None)
        if value is not None:
            dataset_kwargs[knob] = value

    read_kwargs: Dict[str, Any] = {}
    if dataset_kwargs:
        read_kwargs["dataset_kwargs"] = dataset_kwargs
    if args.columns:
        read_kwargs["columns"] = args.columns.split(",")
    if args.memory:
        read_kwargs["memory"] = args.memory
    if args.filesystem_endpoint:
        from pyarrow.fs import S3FileSystem

        read_kwargs["filesystem"] = S3FileSystem(
            endpoint_override=args.filesystem_endpoint,
            scheme="http"
            if args.filesystem_endpoint.startswith("http://")
            else "https",
        )

    sampler = WorkerRssSampler()
    with sampler:
        t0 = time.perf_counter()
        ds = ray.data.read_parquet(args.path, **read_kwargs)
        rows = 0
        if args.consume == "iter_batches":
            # Decodes everything, retains ~one batch: the decode working set with
            # the least retention contamination.
            for batch in ds.iter_batches(batch_format="pyarrow"):
                rows += batch.num_rows
        elif args.consume == "iter_bundles":
            # Retains output blocks in the object store, like the release read_* tests.
            for _ in ds.iter_internal_ref_bundles():
                pass
        elif args.consume == "materialize":
            ds = ds.materialize()
            rows = ds.count()
        else:
            raise ValueError(args.consume)
        wall_s = time.perf_counter() - t0

    result: Dict[str, Any] = {
        "kind": "run",
        "reader": args.reader,
        "shape": args.shape,
        "path": args.path,
        "consume": args.consume,
        "columns": args.columns,
        "wall_s": round(wall_s, 3),
        "rows": rows or None,
        "knobs": dataset_kwargs,
        **sampler.result(),
        **collect_operator_metrics(ds),
    }
    if prof_dir:
        result.update(summarize_profile(prof_dir))
    # operators_detail is verbose; keep it but move the read op up front.
    for op in result.get("operators_detail") or []:
        if "Parquet" in (op.get("operator_name") or "") or "Read" in (
            op.get("operator_name") or ""
        ):
            result["read_op"] = op
            break
    return result


# --------------------------------------------------------------------------- #
# Matrix driver (parent process)
# --------------------------------------------------------------------------- #
def parse_sweep(specs: List[str]) -> List[Dict[str, int]]:
    """``["arrow_rs_k=1,2", "arrow_rs_fetch_window_mb=16,64"]`` -> the cross
    product as a list of knob dicts. Empty input -> one empty dict (defaults)."""
    axes: List[Tuple[str, List[int]]] = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--sweep expects knob=v1,v2 (got {spec!r})")
        knob, raw = spec.split("=", 1)
        if knob not in KNOBS:
            raise SystemExit(f"unknown knob {knob!r}; known: {list(KNOBS)}")
        axes.append((knob, [int(v) for v in raw.split(",") if v != ""]))
    combos: List[Dict[str, int]] = [{}]
    for knob, values in axes:
        combos = [{**c, knob: v} for c in combos for v in values]
    return combos


def child_argv(args, shape: str, reader: str, knobs: Dict[str, int]) -> List[str]:
    path = f"{args.data.rstrip('/')}/{shape}"
    argv = [
        sys.executable,
        os.path.abspath(__file__),
        "--run-one",
        "--path",
        path,
        "--shape",
        shape,
        "--reader",
        reader,
        "--consume",
        args.consume,
    ]
    if args.endpoint:
        argv += ["--filesystem-endpoint", args.endpoint]
    if args.columns:
        argv += ["--columns", args.columns]
    if args.memory:
        argv += ["--memory", str(args.memory)]
    if args.target_block_size_mib:
        argv += ["--target-block-size-mib", str(args.target_block_size_mib)]
    if args.num_cpus:
        argv += ["--num-cpus", str(args.num_cpus)]
    if args.object_store_mb:
        argv += ["--object-store-mb", str(args.object_store_mb)]
    if args.profile:
        argv += ["--prof-dir", os.path.join(args.prof_root, f"{shape}_{reader}")]
    for knob, value in knobs.items():
        argv += [f"--{knob.replace('_', '-')}", str(value)]
    return argv


def fmt_bytes(value: Optional[float]) -> str:
    return f"{value / MiB:.0f}" if value else "-"


def summarize(runs: List[dict]) -> None:
    """Group by (shape, consume, knobs) and print arrow-rs / pyarrow ratios."""
    groups: Dict[Tuple, Dict[str, dict]] = {}
    for run in runs:
        key = (run["shape"], run["consume"], json.dumps(run["knobs"], sort_keys=True))
        groups.setdefault(key, {})[run["reader"]] = run

    header = (
        f"{'shape':16} {'consume':13} {'knobs':34} "
        f"{'wall R/P':>9} {'uss R/P':>8} {'peakRSS R/P':>11} "
        f"{'R uss MiB':>9} {'P uss MiB':>9} {'n':>3} {'flags':<28}"
    )
    print("\n" + header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)
    for (shape, consume, knobs), arms in sorted(groups.items()):
        r, p = arms.get("arrow_rs"), arms.get("pyarrow")

        def ratio(key):
            if not (r and p):
                return None
            a, b = r.get(key), p.get(key)
            return (a / b) if (a and b) else None

        r_uss = (r or {}).get("read_avg_max_uss_per_task_bytes")
        p_uss = (p or {}).get("read_avg_max_uss_per_task_bytes")
        samples = (r or {}).get("read_uss_num_samples")
        flags = []
        if (r or {}).get("prof_oversized_units"):
            flags.append(f"oversized={r['prof_oversized_units']}")
        if (r or {}).get("prof_hstack_row_groups"):
            flags.append(f"hstack={r['prof_hstack_row_groups']}")
        if (r or {}).get("prof_floored_row_groups"):
            flags.append(f"floored={r['prof_floored_row_groups']}")
        if (r or {}).get("prof_fallbacks"):
            flags.append(f"FALLBACK={r['prof_fallbacks']}")
        if (r or {}).get("prof_retained_over_row_group"):
            flags.append(f"retain/rg={r['prof_retained_over_row_group']}")

        def f(x):
            return f"{x:.2f}x" if x else "-"

        print(
            f"{shape:16} {consume:13} {knobs[:34]:34} "
            f"{f(ratio('wall_s')):>9} {f(ratio('read_avg_max_uss_per_task_bytes')):>8} "
            f"{f(ratio('worker_peak_sum_rss_bytes')):>11} "
            f"{fmt_bytes(r_uss):>9} {fmt_bytes(p_uss):>9} "
            f"{(samples if samples is not None else '-'):>3} {' '.join(flags):<28}",
            file=sys.stderr,
        )
    print(
        "\nuss = per-read-task avg peak USS (Linux only; '-' means 0 samples, i.e. "
        "unmeasured, NOT flat).\npeakRSS = summed RSS across ray workers at its "
        "worst instant (upper bound; shared pages double-counted).",
        file=sys.stderr,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)

    # Matrix mode
    p.add_argument("--data", help="Fixture root: local dir or s3://bucket/prefix")
    p.add_argument("--shapes", default="fat_col,fat_col_solo,narrow_big_rg,fat_rows")
    p.add_argument("--readers", default="arrow_rs,pyarrow")
    p.add_argument(
        "--sweep",
        action="append",
        default=[],
        help="knob=v1,v2 (repeatable; cross product). " f"Knobs: {', '.join(KNOBS)}",
    )
    p.add_argument("--repeat", type=int, default=1, help="Runs per configuration")
    p.add_argument("--profile", action="store_true", default=True)
    p.add_argument("--no-profile", dest="profile", action="store_false")
    p.add_argument("--prof-root", default="/tmp/arrow_rs_prof")

    # Shared
    p.add_argument(
        "--consume",
        default="iter_batches",
        choices=["iter_batches", "iter_bundles", "materialize"],
    )
    p.add_argument("--columns", default=None, help="Comma-separated projection")
    p.add_argument("--memory", type=int, default=None, help="read_parquet(memory=)")
    p.add_argument("--target-block-size-mib", type=int, default=None)
    p.add_argument("--endpoint", default=None, help="S3 endpoint (MinIO/moto)")
    p.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Pin ray.init(num_cpus=...) so both arms get the same parallelism",
    )
    p.add_argument(
        "--object-store-mb",
        type=int,
        default=None,
        help="Pin ray.init(object_store_memory=...); default derives from free RAM",
    )

    # Child-only
    p.add_argument("--path")
    p.add_argument("--shape")
    p.add_argument("--reader", choices=["arrow_rs", "pyarrow"])
    p.add_argument("--prof-dir", default=None)
    p.add_argument("--filesystem-endpoint", default=None)
    for knob in KNOBS:
        p.add_argument(f"--{knob.replace('_', '-')}", type=int, default=None)

    args = p.parse_args(argv)
    if not args.run_one and not args.data:
        p.error("--data is required")
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.run_one:
        print(json.dumps(run_one(args)))
        return

    combos = parse_sweep(args.sweep)
    shapes = [s for s in args.shapes.split(",") if s]
    readers = [r for r in args.readers.split(",") if r]

    runs: List[dict] = []
    for shape in shapes:
        for knobs in combos:
            for reader in readers:
                # Knobs only exist on the arrow-rs path; running PyArrow once per
                # knob combo would just re-measure the same thing. Reuse its first
                # result as the shared baseline for every combo of this shape.
                if (
                    reader == "pyarrow"
                    and knobs
                    and any(
                        r["shape"] == shape and r["reader"] == "pyarrow" for r in runs
                    )
                ):
                    prior = next(
                        r
                        for r in runs
                        if r["shape"] == shape and r["reader"] == "pyarrow"
                    )
                    runs.append({**prior, "knobs": knobs, "reused_baseline": True})
                    continue
                for attempt in range(args.repeat):
                    prof_dir = os.path.join(args.prof_root, f"{shape}_{reader}")
                    if args.profile:
                        shutil.rmtree(prof_dir, ignore_errors=True)
                    argv_child = child_argv(args, shape, reader, knobs)
                    label = f"{shape} {reader} {knobs or 'defaults'}"
                    print(f"--- run {label} (attempt {attempt + 1})", file=sys.stderr)
                    proc = subprocess.run(
                        argv_child, capture_output=True, text=True, check=False
                    )
                    if proc.returncode != 0:
                        print(
                            f"FAILED {label}\n{proc.stderr[-3000:]}",
                            file=sys.stderr,
                        )
                        continue
                    record = None
                    for line in proc.stdout.splitlines():
                        line = line.strip()
                        if line.startswith("{"):
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                pass
                    if record is None:
                        print(f"NO RESULT {label}", file=sys.stderr)
                        continue
                    record["attempt"] = attempt
                    runs.append(record)
                    print(json.dumps(record))

    # Median across repeats, so a summary line is not one noisy run.
    if args.repeat > 1:
        merged: Dict[Tuple, List[dict]] = {}
        for run in runs:
            key = (
                run["shape"],
                run["reader"],
                run["consume"],
                json.dumps(run["knobs"], sort_keys=True),
            )
            merged.setdefault(key, []).append(run)
        runs = []
        for group in merged.values():
            base = dict(group[0])
            for key in ("wall_s", "worker_peak_sum_rss_bytes"):
                vals = [g[key] for g in group if g.get(key)]
                if vals:
                    base[key] = statistics.median(vals)
            base["repeats"] = len(group)
            runs.append(base)

    summarize(runs)


if __name__ == "__main__":
    main()
