"""Detailed benchmark instrumentation for TQ benefit quantification.

Usage:
    from agentlightning.bench_detail import bench_log, now, rss_mb

    t0 = now()
    # ... operation ...
    t1 = now()
    bench_log("main", step, "data_transfer", query_spans=t1-t0)

Environment variables:
    BENCH_DETAIL: "1" to enable (default), "0" to disable.
"""
import json
import os
import socket
import time

_BENCH_DIR = "/home/ma-user/install"
_BENCH_ENABLED = os.environ.get("BENCH_DETAIL", "1") == "1"
_HOSTNAME = socket.gethostname()


def now():
    return time.perf_counter()


def rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0.0


def bench_log(scheme: str, step: int, category: str, **metrics):
    if not _BENCH_ENABLED:
        return
    entry = {
        "scheme": scheme,
        "step": step,
        "category": category,
        "ts": time.time(),
        "host": _HOSTNAME,
        **metrics,
    }
    line = json.dumps(entry, default=str)
    path = os.path.join(_BENCH_DIR, f"bench_detail_{scheme}.jsonl")
    try:
        os.makedirs(_BENCH_DIR, exist_ok=True)
        with open(path, "a") as f:
            f.write(line + "\n")
    except Exception:
        print(f"[BENCH_DETAIL] {line}", flush=True)
