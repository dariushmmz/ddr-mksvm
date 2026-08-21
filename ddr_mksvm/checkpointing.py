"""Crash-safe, per-run checkpoints for the DDR-MKSVM experiment drivers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile

import numpy as np


CHECKPOINT_VERSION = 1


def file_sha256(path, chunk_size=1024 * 1024):
    """Return a stable fingerprint used to reject checkpoints for other data."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def save_run_checkpoint(path, run_index, seed, values):
    """Atomically save one completed run; safe for separate joblib workers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(fd)
    try:
        np.savez_compressed(
            temporary_name,
            version=np.array(CHECKPOINT_VERSION, dtype=np.int64),
            run_index=np.array(run_index, dtype=np.int64),
            seed=np.array(seed, dtype=np.uint32),
            values=np.asarray(values, dtype=float).reshape(-1),
        )
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def run_and_checkpoint(work_function, data, work_kwargs, run_index, seed,
                       checkpoint_path):
    """Execute one experiment run and save its returned metric(s)."""
    result = work_function(data, seed=int(seed), **work_kwargs)
    values = np.asarray(result, dtype=float).reshape(-1)
    save_run_checkpoint(checkpoint_path, run_index, seed, values)
    return run_index, values


class RunCheckpointStore:
    """Manage the manifest and individual result files for one ablation."""

    def __init__(self, directory, metadata, n_runs, seeded, value_count,
                 resume=False):
        self.directory = Path(directory)
        self.manifest_path = self.directory / "manifest.json"
        self.n_runs = int(n_runs)
        self.value_count = int(value_count)
        self.metadata = dict(metadata)
        self.metadata.update({
            "version": CHECKPOINT_VERSION,
            "n_runs": self.n_runs,
            "seeded": bool(seeded),
            "value_count": self.value_count,
        })

        if self.n_runs <= 0:
            raise ValueError("n_runs must be a positive integer")

        self.directory.mkdir(parents=True, exist_ok=True)
        if resume:
            self.manifest = self._load_or_create_manifest(seeded)
        else:
            self._clear_run_files()
            self.manifest = self._new_manifest(seeded)
            _atomic_json_write(self.manifest_path, self.manifest)

        self.seeds = [int(seed) for seed in self.manifest["seeds"]]

    def _new_manifest(self, seeded):
        if seeded:
            seeds = list(range(self.n_runs))
        else:
            seeds = [secrets.randbelow(2 ** 32) for _ in range(self.n_runs)]
        return {**self.metadata, "seeds": seeds}

    def _load_or_create_manifest(self, seeded):
        if not self.manifest_path.exists():
            manifest = self._new_manifest(seeded)
            _atomic_json_write(self.manifest_path, manifest)
            return manifest

        try:
            with self.manifest_path.open("r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot resume because the checkpoint manifest is unreadable: "
                f"{self.manifest_path}"
            ) from exc

        mismatches = []
        for key, expected in self.metadata.items():
            if manifest.get(key) != expected:
                mismatches.append(
                    f"{key}: checkpoint={manifest.get(key)!r}, requested={expected!r}"
                )
        seeds = manifest.get("seeds")
        if not isinstance(seeds, list) or len(seeds) != self.n_runs:
            mismatches.append("the stored seed schedule is missing or has the wrong length")
        if mismatches:
            details = "; ".join(mismatches)
            raise ValueError(
                f"Checkpoint configuration does not match this run ({details}). "
                "Run without --resume to start a fresh checkpoint."
            )
        return manifest

    def _clear_run_files(self):
        for path in self.directory.glob("run_*.npz"):
            path.unlink()
        try:
            self.manifest_path.unlink()
        except FileNotFoundError:
            pass

    def run_path(self, run_index):
        return self.directory / f"run_{run_index:05d}.npz"

    def load_completed(self):
        """Return valid completed results; corrupt/partial files are rerun."""
        completed = {}
        for run_index, seed in enumerate(self.seeds):
            path = self.run_path(run_index)
            if not path.exists():
                continue
            try:
                with np.load(path, allow_pickle=False) as checkpoint:
                    version = int(checkpoint["version"])
                    saved_index = int(checkpoint["run_index"])
                    saved_seed = int(checkpoint["seed"])
                    values = np.asarray(checkpoint["values"], dtype=float).reshape(-1)
                if (version != CHECKPOINT_VERSION or saved_index != run_index or
                        saved_seed != seed or len(values) != self.value_count or
                        not np.all(np.isfinite(values))):
                    raise ValueError("checkpoint contents do not match the manifest")
            except (OSError, KeyError, ValueError) as exc:
                print(f"WARNING: Ignoring invalid checkpoint {path}: {exc}")
                continue
            completed[run_index] = values
        return completed
