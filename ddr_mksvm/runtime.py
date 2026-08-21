"""Runtime resource detection for notebook/container experiment drivers."""

import math
import os


def _read_cpu_quota(path, period_path=None):
    """Read a cgroup v2 cpu.max file or the paired cgroup v1 files."""
    try:
        if period_path is None:
            quota_text, period_text = open(path, encoding="ascii").read().split()[:2]
        else:
            quota_text = open(path, encoding="ascii").read().strip()
            period_text = open(period_path, encoding="ascii").read().strip()
        if quota_text == "max":
            return None
        quota, period = int(quota_text), int(period_text)
        if quota <= 0 or period <= 0:
            return None
        return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        return None


def effective_cpu_count():
    """Best-effort CPU count that respects affinity and Linux cgroup quotas."""
    candidates = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        try:
            candidates.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass

    quota = _read_cpu_quota("/sys/fs/cgroup/cpu.max")
    if quota is None:
        quota = _read_cpu_quota(
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        )
    if quota is not None:
        candidates.append(quota)
    return max(1, min(candidates))


def resolve_n_jobs(requested, n_tasks):
    """Resolve the CLI value without passing unsafe ``-1`` to joblib."""
    if n_tasks <= 0:
        raise ValueError("n_tasks must be positive")
    if requested is None or requested == -1:
        workers = effective_cpu_count()
    elif requested <= 0:
        raise ValueError("n_jobs must be -1 or a positive integer")
    else:
        workers = requested
    return min(workers, n_tasks)
