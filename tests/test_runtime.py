import pytest

import ddr_mksvm.runtime as runtime


def test_resolve_n_jobs_uses_detected_limit(monkeypatch):
    monkeypatch.setattr(runtime, "effective_cpu_count", lambda: 16)
    assert runtime.resolve_n_jobs(None, 96) == 16
    assert runtime.resolve_n_jobs(-1, 96) == 16
    assert runtime.resolve_n_jobs(8, 96) == 8
    assert runtime.resolve_n_jobs(32, 10) == 10


def test_resolve_n_jobs_rejects_zero():
    with pytest.raises(ValueError, match="n_jobs"):
        runtime.resolve_n_jobs(0, 96)
