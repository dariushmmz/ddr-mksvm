import json

import numpy as np
import pytest

from ddr_mksvm.checkpointing import (
    RunCheckpointStore,
    run_and_checkpoint,
    save_run_checkpoint,
)
from main_ddr_binary import ABLATIONS, load_ablation_from_checkpoints


def _fake_work(data, seed, offset):
    return float(data + offset), float(seed % 17)


def test_completed_runs_are_loaded_on_resume(tmp_path):
    metadata = {"kind": "test", "dataset": "sample", "ablation": "full"}
    store = RunCheckpointStore(
        tmp_path, metadata, n_runs=3, seeded=False, value_count=2
    )

    run_index, values = run_and_checkpoint(
        _fake_work, 2.0, {"offset": 3.0}, 1, store.seeds[1], store.run_path(1)
    )
    assert run_index == 1
    np.testing.assert_allclose(values[0], 5.0)

    resumed = RunCheckpointStore(
        tmp_path, metadata, n_runs=3, seeded=False, value_count=2, resume=True
    )
    assert resumed.seeds == store.seeds
    completed = resumed.load_completed()
    assert set(completed) == {1}
    np.testing.assert_allclose(completed[1], values)


def test_seeded_manifest_uses_run_indices(tmp_path):
    store = RunCheckpointStore(
        tmp_path, {"kind": "test"}, n_runs=4, seeded=True, value_count=1
    )
    assert store.seeds == [0, 1, 2, 3]


def test_resume_rejects_configuration_changes(tmp_path):
    RunCheckpointStore(
        tmp_path, {"dataset_sha256": "old"}, n_runs=2,
        seeded=True, value_count=1,
    )
    with pytest.raises(ValueError, match="does not match"):
        RunCheckpointStore(
            tmp_path, {"dataset_sha256": "new"}, n_runs=2,
            seeded=True, value_count=1, resume=True,
        )


def test_invalid_run_checkpoint_is_ignored(tmp_path):
    store = RunCheckpointStore(
        tmp_path, {"kind": "test"}, n_runs=1, seeded=True, value_count=1
    )
    store.run_path(0).write_bytes(b"partial checkpoint")
    assert store.load_completed() == {}


def test_fresh_store_removes_old_run_files(tmp_path):
    metadata = {"kind": "test"}
    old = RunCheckpointStore(
        tmp_path, metadata, n_runs=1, seeded=True, value_count=1
    )
    save_run_checkpoint(old.run_path(0), 0, 0, [0.25])
    assert old.run_path(0).exists()

    fresh = RunCheckpointStore(
        tmp_path, metadata, n_runs=1, seeded=True, value_count=1
    )
    assert not fresh.run_path(0).exists()
    with fresh.manifest_path.open(encoding="utf-8") as stream:
        assert json.load(stream)["seeds"] == [0]


def test_summary_loader_uses_seed_mode_and_run_count_from_manifest(tmp_path):
    metadata = {
        "kind": "binary",
        "dataset": "sample",
        "ablation": "full",
        "flags": ABLATIONS["full"],
        "dataset_sha256": "abc123",
    }
    store = RunCheckpointStore(
        tmp_path, metadata, n_runs=2, seeded=True, value_count=3
    )
    save_run_checkpoint(store.run_path(0), 0, 0, [0.25, 0.1, 0.2])
    save_run_checkpoint(store.run_path(1), 1, 1, [0.75, 0.3, 0.4])

    result = load_ablation_from_checkpoints(
        list(range(8)), "sample", "full", tmp_path, "abc123"
    )

    assert result["mean"] == pytest.approx(0.5)
    assert result["std_empirical"] == pytest.approx(0.25)
    np.testing.assert_allclose(result["testing_error_A"], [0.1, 0.3])


def test_summary_loader_rejects_incomplete_checkpoint(tmp_path):
    metadata = {
        "kind": "binary",
        "dataset": "sample",
        "ablation": "dro_only",
        "flags": ABLATIONS["dro_only"],
        "dataset_sha256": "abc123",
    }
    store = RunCheckpointStore(
        tmp_path, metadata, n_runs=2, seeded=False, value_count=3
    )
    save_run_checkpoint(store.run_path(0), 0, store.seeds[0], [0.2, 0.1, 0.1])

    with pytest.raises(ValueError, match="incomplete"):
        load_ablation_from_checkpoints(
            list(range(8)), "sample", "dro_only", tmp_path, "abc123"
        )
