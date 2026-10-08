"""Isolated adapters for the frozen-model expanded dataset study.

This module does not change V7, V8-C, or the q1 Legacy primitives.  It adds a
leakage-safe categorical representation for the local Breast Cancer recurrence
dataset and composes the existing frozen functions for Dermatology.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

from . import v7_cross_dataset as frozen
from . import v8_class_sensitive as v8
from .iris_research import ALPHA_RULES, PAPER_NU_GRID, SVC_C_GRID, alpha_from_rule


DATASETS = {
    "breast_cancer_recurrence": {
        "file": "breast_cancer.csv",
        "kind": "categorical",
        "transform": "standardization",
        "kernel": "hom_linear",
        "source_profile_only": True,
    },
    "dermatology": {
        "file": "dermatology.csv",
        "kind": "numeric_frozen",
        "transform": "none",
        "kernel": "inhom_quadratic",
        "complete_case": True,
    },
}

MODELS = ("legacy", "v7", "v8_c")
BREAST_FEATURE_NAMES = (
    "age", "menopause", "tumor_size", "inv_nodes", "node_caps",
    "deg_malig", "breast", "breast_quad", "irradiat",
)
TOKEN_REPAIRS = {
    2: {"9-May": "5-9", "14-Oct": "10-14"},
    3: {"5-Mar": "3-5", "8-Jun": "6-8", "11-Sep": "9-11", "14-Dec": "12-14"},
}
PREPROCESSING_DEFINITION = {
    "breast_cancer_recurrence": {
        "repairs": {str(k): v for k, v in TOKEN_REPAIRS.items()},
        "missing_token": "__MISSING__",
        "encoding": "OneHotEncoder fit on current training partition; handle_unknown=ignore; dense float64",
        "scaling": "population standardization fit after encoding on current training partition",
        "legacy_profile": "standardization + homogeneous linear q1",
    },
    "dermatology": {
        "rows": "complete case using frozen loader",
        "scaling": "none",
        "legacy_profile": "none + inhomogeneous quadratic q1",
    },
}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def preprocessing_definition_hash() -> str:
    return _canonical_hash(PREPROCESSING_DEFINITION)


def repair_breast_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore unambiguous interval tokens and preserve missing as a category."""
    if frame.shape[1] != 9:
        raise ValueError("Breast Cancer recurrence must have nine predictors")
    out = frame.copy().astype("string")
    out.columns = list(BREAST_FEATURE_NAMES)
    for col_index, mapping in TOKEN_REPAIRS.items():
        out.iloc[:, col_index] = out.iloc[:, col_index].replace(mapping)
    out = out.fillna("__MISSING__")
    for col in out.columns:
        out[col] = out[col].str.strip().replace("", "__MISSING__")
    return out


def load_dataset(name: str, root: str = "dataset"):
    if name not in DATASETS:
        raise KeyError(name)
    spec = dict(DATASETS[name])
    path = Path(root) / spec["file"]
    if name == "dermatology":
        X, y, base = frozen.load_dataset(name, root)
        spec.update(base)
        spec["preprocessing_definition_hash"] = preprocessing_definition_hash()
        return X, y, spec
    raw = pd.read_csv(path)
    X = repair_breast_features(raw.iloc[:, :-1])
    y = pd.to_numeric(raw.iloc[:, -1], errors="raise").to_numpy(int)
    classes, counts = np.unique(y, return_counts=True)
    if classes.tolist() != [0, 1]:
        raise ValueError(f"unexpected labels {classes.tolist()}")
    cleaned_payload = X.to_csv(index=False, lineterminator="\n").encode() + y.astype("<i8").tobytes()
    spec.update({
        "csv_sha256": frozen.sha256_file(path),
        "semantic_data_sha256": _sha_bytes(cleaned_payload),
        "raw_rows": len(raw),
        "rows": len(y),
        "features": 9,
        "classes": classes.tolist(),
        "class_counts": {str(c): int(n) for c, n in zip(classes, counts)},
        "missing_as_category": {"node_caps": 8, "breast_quad": 1},
        "preprocessing_definition_hash": preprocessing_definition_hash(),
    })
    return X, y, spec


def _encode_scale(A: pd.DataFrame, B: pd.DataFrame):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    EA = encoder.fit_transform(A)
    EB = encoder.transform(B)
    SA, SB, scaling = frozen.fit_transform(EA, EB, "standardization")
    categories = [[str(x) for x in values.tolist()] for values in encoder.categories_]
    metadata = {
        "input_features": list(A.columns),
        "categories": categories,
        "encoded_features": encoder.get_feature_names_out(A.columns).tolist(),
        "encoded_dimension": int(EA.shape[1]),
        "handle_unknown": "ignore",
        "standardization": scaling,
    }
    metadata["preprocessing_hash"] = _canonical_hash(metadata)
    return SA, SB, metadata


def _fit_v7_categorical(X: pd.DataFrame, y: np.ndarray, Z: pd.DataFrame, seed: int):
    started = time.perf_counter()
    folds = list(StratifiedKFold(3, shuffle=True, random_state=20000 + int(seed)).split(X, y))
    prepared = []
    fold_meta = []
    for fold, (fi, va) in enumerate(folds):
        A, B, meta = _encode_scale(X.iloc[fi], X.iloc[va])
        prepared.append((A, B, y[fi], y[va]))
        fold_meta.append({"fold": fold, "train_indices": fi.tolist(), "validation_indices": va.tolist(), "preprocessing": meta})
    candidates = []
    diagnostics = []
    for ai, rule in enumerate(ALPHA_RULES):
        for ci, C in enumerate(SVC_C_GRID):
            errors = []
            for fold, (A, B, ya, yb) in enumerate(prepared):
                alpha = alpha_from_rule(A, rule)
                cl = SVC(C=C, kernel="rbf", gamma=1 / (2 * alpha**2),
                         decision_function_shape="ovo", shrinking=True, tol=1e-9).fit(A, ya)
                pred = cl.predict(B)
                error = float(np.mean(pred != yb))
                errors.append(error)
                diagnostics.append({
                    "model": "v7", "phase": "inner", "fold": fold,
                    "alpha_rule": rule, "alpha": alpha, "C": float(C),
                    "validation_error": error, "solver": "libsvm-smo",
                    "solver_status": "ok" if cl.fit_status_ == 0 else "failed",
                    "solver_iterations": int(np.sum(cl.n_iter_)), "selected": False,
                })
            candidates.append((float(np.mean(errors)), ai, ci, rule, float(C)))
    best = min(candidates)
    rule, C = best[3], best[4]
    A, B, outer_meta = _encode_scale(X, Z)
    alpha = alpha_from_rule(A, rule)
    cl = SVC(C=C, kernel="rbf", gamma=1 / (2 * alpha**2),
             decision_function_shape="ovo", shrinking=True, tol=1e-9).fit(A, y)
    pred = cl.predict(B)
    diagnostics.append({
        "model": "v7", "phase": "outer", "fold": -1,
        "alpha_rule": rule, "alpha": alpha, "C": C,
        "validation_error": best[0], "solver": "libsvm-smo",
        "solver_status": "ok" if cl.fit_status_ == 0 else "failed",
        "solver_iterations": int(np.sum(cl.n_iter_)), "selected": True,
        "support_vectors": int(cl.support_.size),
    })
    params = {
        "alpha_rule": rule, "alpha": alpha, "C": C,
        "inner_error": best[0], "preprocessing": outer_meta,
        "support_vectors": int(cl.support_.size),
    }
    return pred, diagnostics, time.perf_counter() - started, params, fold_meta


def _fit_v8c_categorical(X: pd.DataFrame, y: np.ndarray, Z: pd.DataFrame, seed: int):
    started = time.perf_counter()
    folds = list(StratifiedKFold(3, shuffle=True, random_state=20000 + int(seed)).split(X, y))
    prepared = []
    fold_meta = []
    for fold, (fi, va) in enumerate(folds):
        A, B, meta = _encode_scale(X.iloc[fi], X.iloc[va])
        prepared.append((A, B, y[fi], y[va]))
        fold_meta.append({"fold": fold, "train_indices": fi.tolist(), "validation_indices": va.tolist(), "preprocessing": meta})
    candidates = []
    inner_diagnostics = []
    for ai, rule in enumerate(ALPHA_RULES):
        for ci, C in enumerate(SVC_C_GRID):
            fold_losses = []
            for fold, (A, B, ya, yb) in enumerate(prepared):
                alpha = alpha_from_rule(A, rule)
                model = v8.fit_model(A, ya, float(C), alpha, "sqrt")
                pred, _, _ = v8.predict_model(model, B)
                loss = v8.losses(yb, pred)
                fold_losses.append(loss)
                for diag in model["diagnostics"]:
                    inner_diagnostics.append({
                        "model": "v8_c", "phase": "inner", "fold": fold,
                        "alpha_rule": rule, "alpha": alpha, "C": float(C),
                        "validation_balanced_loss": loss["balanced"],
                        "validation_error": loss["error"],
                        "validation_macro_f1_loss": loss["f1"],
                        **diag,
                    })
            means = {key: float(np.mean([x[key] for x in fold_losses])) for key in ("error", "balanced", "f1")}
            candidates.append((means["balanced"], ai, ci, rule, float(C), means))
    selected = min(candidates)
    rule, C, losses = selected[3], selected[4], selected[5]
    A, B, outer_meta = _encode_scale(X, Z)
    alpha = alpha_from_rule(A, rule)
    model = v8.fit_model(A, y, C, alpha, "sqrt")
    pred, margins, votes = v8.predict_model(model, B)
    outer_diagnostics = []
    for diag in model["diagnostics"]:
        outer_diagnostics.append({
            "model": "v8_c", "phase": "outer", "fold": -1,
            "alpha_rule": rule, "alpha": alpha, "C": C,
            "validation_balanced_loss": losses["balanced"],
            "validation_error": losses["error"],
            "validation_macro_f1_loss": losses["f1"],
            "selected": True, **diag,
        })
    params = {
        "alpha_rule": rule, "alpha": alpha, "C": C, "ratio": 1.0,
        "losses": losses, "order": [selected[1], selected[2], 0],
        "family": "sqrt", "selection_metric": "balanced",
        "preprocessing": outer_meta, "pairs": model["diagnostics"],
        "support_vectors": int(sum(d["support_vectors"] for d in model["diagnostics"])),
    }
    trace = {
        "margins_positive_second": margins.tolist(), "votes": votes.tolist(),
        "vote_ties": int(np.sum((votes == votes.max(axis=1)[:, None]).sum(axis=1) > 1)),
    }
    return pred, inner_diagnostics + outer_diagnostics, time.perf_counter() - started, params, fold_meta, trace


def _augment_record(record: dict, y_train: np.ndarray, y_test: np.ndarray, pred: np.ndarray):
    classes, counts = np.unique(y_train, return_counts=True)
    rare = classes[np.argmin(counts)]
    labels = json.loads(record["labels"])
    recalls = json.loads(record["class_recall"])
    record["rare_class"] = int(rare)
    record["rare_class_recall"] = float(recalls[str(int(rare))])
    record["majority_only"] = bool(len(np.unique(pred)) == 1 and len(labels) > 1)
    record["true_label_hash"] = _sha_bytes(np.asarray(y_test, dtype="<i8").tobytes())
    params = json.loads(record["selected_hyperparameters"])
    prep = params.get("preprocessing", {})
    record["preprocessing_hash"] = prep.get("preprocessing_hash", _canonical_hash(prep))
    return record


def _evaluate_categorical(name: str, X: pd.DataFrame, y: np.ndarray, spec: dict, seed: int, models):
    tr, te = frozen.split_indices(y, seed)
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    predictions, params, runtimes, solver = {}, {}, {}, []
    folds, traces = {}, {}
    if "legacy" in models:
        A, B, legacy_prep = _encode_scale(Xtr, Xte)
        legacy_pred, selected_indices, legacy_diag, legacy_runtime, legacy_params = frozen._legacy_fit_predict(A, y[tr], B, spec, seed)
        legacy_selected = [legacy_diag[i].get("nu") for i in selected_indices]
        legacy_params.update({
            "preprocessing": legacy_prep, "nu_grid": list(PAPER_NU_GRID),
            "selected_nu": legacy_selected, "source_profile_only": True,
        })
        for row in legacy_diag:
            row.update(dataset=name, model="legacy", seed=seed)
        predictions["legacy"], params["legacy"], runtimes["legacy"] = legacy_pred, legacy_params, legacy_runtime
        solver.extend(legacy_diag)
    if "v7" in models:
        v7_pred, v7_diag, v7_runtime, v7_params, v7_folds = _fit_v7_categorical(Xtr, y[tr], Xte, seed)
        for row in v7_diag:
            row.update(dataset=name, seed=seed)
        predictions["v7"], params["v7"], runtimes["v7"] = v7_pred, v7_params, v7_runtime
        solver.extend(v7_diag)
        folds["v7"] = v7_folds
    if "v8_c" in models:
        c_pred, c_diag, c_runtime, c_params, c_folds, c_trace = _fit_v8c_categorical(Xtr, y[tr], Xte, seed)
        for row in c_diag:
            row.update(dataset=name, seed=seed)
        predictions["v8_c"], params["v8_c"], runtimes["v8_c"] = c_pred, c_params, c_runtime
        solver.extend(c_diag)
        folds["v8_c"] = c_folds
        traces["v8_c"] = c_trace
    records = []
    for model in models:
        rec = frozen.metric_record(name, model, seed, y[te], predictions[model], runtimes[model], params[model], tr, te)
        records.append(_augment_record(rec, y[tr], y[te], predictions[model]))
    return {
        "records": records,
        "solver_diagnostics": solver,
        "split": {"dataset": name, "seed": seed, "train_indices": tr.tolist(), "test_indices": te.tolist(), "y_test": y[te].tolist()},
        "preprocessing_folds": folds,
        "traces": traces,
    }


def _evaluate_numeric(name: str, X: np.ndarray, y: np.ndarray, spec: dict, seed: int, models):
    tr, te = frozen.split_indices(y, seed)
    records, solver, folds, traces = [], [], {}, {}
    if "legacy" in models:
        A, B, prep = frozen.fit_transform(X[tr], X[te], spec["transform"])
        legacy_pred, selected_indices, legacy_diag, legacy_runtime, legacy_params = frozen._legacy_fit_predict(A, y[tr], B, spec, seed)
        legacy_params.update({
            "preprocessing": prep, "nu_grid": list(PAPER_NU_GRID),
            "selected_nu": [legacy_diag[i].get("nu") for i in selected_indices],
        })
        legacy_rec = frozen.metric_record(name, "legacy", seed, y[te], legacy_pred, legacy_runtime, legacy_params, tr, te)
        records.append(_augment_record(legacy_rec, y[tr], y[te], legacy_pred))
        for row in legacy_diag:
            row.update(dataset=name, model="legacy", seed=seed)
        solver.extend(legacy_diag)
    variants = [model for model in ("v7", "v8_c") if model in models]
    if variants:
        v8_result = v8.evaluate_variants(name, X, y, spec, seed, variants)
        if v8_result["split"]["train_indices"] != tr.tolist() or v8_result["split"]["test_indices"] != te.tolist():
            raise AssertionError("frozen split mismatch")
        for rec in v8_result["records"]:
            pred = np.asarray(json.loads(rec["prediction_vector"]), int)
            records.append(_augment_record(rec, y[tr], y[te], pred))
        for model, trace in v8_result["traces"].items():
            traces[model] = trace
            for row in trace.get("solver_diagnostics", []):
                solver.append({"dataset": name, "model": model, "seed": seed, "phase": "outer", **row})
        for family, bank in v8_result["banks"].items():
            folds[family] = bank["folds"]
            for candidate in bank["records"]:
                for fold in candidate["folds"]:
                    for row in fold["solver_diagnostics"]:
                        solver.append({
                            "dataset": name, "model": "v8_c", "seed": seed, "phase": "inner",
                            "fold": fold["fold"], "alpha_rule": candidate["alpha_rule"],
                            "C": candidate["C"], "validation_balanced_loss": fold["losses"]["balanced"], **row,
                        })
    return {
        "records": records,
        "solver_diagnostics": solver,
        "split": {"dataset": name, "seed": seed, "train_indices": tr.tolist(), "test_indices": te.tolist(), "y_test": y[te].tolist()},
        "preprocessing_folds": folds,
        "traces": traces,
    }


def evaluate_seed(name: str, X, y: np.ndarray, spec: dict, seed: int, models=MODELS):
    models = tuple(models)
    if not models or any(model not in MODELS for model in models):
        raise ValueError(f"invalid model subset {models}")
    if "v8_c" in models and "v7" not in models:
        raise ValueError("v8_c requires the matched v7 reference in the same job")
    started = time.perf_counter()
    result = _evaluate_categorical(name, X, y, spec, seed, models) if spec["kind"] == "categorical" else _evaluate_numeric(name, X, y, spec, seed, models)
    result["actual_job_wall_s"] = time.perf_counter() - started
    return result


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), group in frame.groupby(["dataset", "model"]):
        row = {"dataset": dataset, "model": model, "seeds": len(group)}
        for metric in ("test_error", "balanced_accuracy", "macro_f1", "rare_class_recall", "runtime_s"):
            values = group[metric].to_numpy(float)
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"median_{metric}"] = float(np.median(values))
            row[f"sd_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        row["majority_only_count"] = int(group["majority_only"].astype(bool).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in frame.groupby("dataset"):
        for reference, candidate in (("legacy", "v7"), ("v7", "v8_c")):
            ref = group[group.model == reference].set_index("seed")
            cand = group[group.model == candidate].set_index("seed")
            common = sorted(set(ref.index) & set(cand.index))
            if not common:
                continue
            ref, cand = ref.loc[common], cand.loc[common]
            for field in ("train_indices_hash", "test_indices_hash", "true_label_hash"):
                if not (ref[field] == cand[field]).all():
                    raise AssertionError(f"unmatched {field}")
            for metric in ("test_error", "balanced_accuracy", "macro_f1", "rare_class_recall", "runtime_s"):
                delta = (cand[metric] - ref[metric]).to_numpy(float)
                n = len(delta)
                sd = float(np.std(delta, ddof=1)) if n > 1 else 0.0
                half = float(t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else None
                favorable = -delta if metric in ("test_error", "runtime_s") else delta
                nonzero = delta[np.abs(delta) > 1e-14]
                rows.append({
                    "dataset": dataset, "reference": reference, "candidate": candidate,
                    "metric": metric, "seeds": n, "mean_reference": float(ref[metric].mean()),
                    "mean_candidate": float(cand[metric].mean()), "mean_delta": float(np.mean(delta)),
                    "ci_low": float(np.mean(delta) - half) if half is not None else None,
                    "ci_high": float(np.mean(delta) + half) if half is not None else None,
                    "wins": int((favorable > 1e-14).sum()), "ties": int((np.abs(favorable) <= 1e-14).sum()),
                    "losses": int((favorable < -1e-14).sum()),
                    "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0,
                    "standardized_paired_effect": float(np.mean(delta) / sd) if sd else None,
                    "runtime_ratio": float(cand.runtime_s.sum() / ref.runtime_s.sum()),
                })
    return pd.DataFrame(rows)


def class_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), group in frame.groupby(["dataset", "model"]):
        labels = json.loads(group.iloc[0].labels)
        matrix = sum(np.asarray(json.loads(value), int) for value in group.confusion_matrix)
        for index, label in enumerate(labels):
            tp = int(matrix[index, index])
            precision = tp / matrix[:, index].sum() if matrix[:, index].sum() else 0.0
            recall = tp / matrix[index].sum() if matrix[index].sum() else 0.0
            rows.append({
                "dataset": dataset, "model": model, "label": label,
                "support": int(matrix[index].sum()), "precision": precision,
                "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "confusion_row": json.dumps(matrix[index].tolist()),
            })
    return pd.DataFrame(rows)


def paired_class_recall(frame: pd.DataFrame) -> pd.DataFrame:
    """Paired per-seed class-recall inference for every available comparison."""
    rows = []
    for dataset, group in frame.groupby("dataset"):
        for reference, candidate in (("legacy", "v7"), ("v7", "v8_c")):
            ref = group[group.model == reference].set_index("seed")
            cand = group[group.model == candidate].set_index("seed")
            common = sorted(set(ref.index) & set(cand.index))
            if not common:
                continue
            ref, cand = ref.loc[common], cand.loc[common]
            if not (ref["true_label_hash"] == cand["true_label_hash"]).all():
                raise AssertionError("unmatched labels for paired class recall")
            labels = sorted({
                str(label)
                for value in list(ref.class_recall) + list(cand.class_recall)
                for label in json.loads(value)
            })
            for label in labels:
                a = np.asarray([json.loads(value)[label] for value in ref.class_recall], float)
                b = np.asarray([json.loads(value)[label] for value in cand.class_recall], float)
                delta = b - a
                n = len(delta)
                sd = float(np.std(delta, ddof=1)) if n > 1 else 0.0
                half = float(t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else None
                nonzero = delta[np.abs(delta) > 1e-14]
                rows.append({
                    "dataset": dataset, "reference": reference, "candidate": candidate,
                    "label": label, "seeds": n,
                    "mean_reference": float(a.mean()), "mean_candidate": float(b.mean()),
                    "mean_delta": float(delta.mean()),
                    "ci_low": float(delta.mean() - half) if half is not None else None,
                    "ci_high": float(delta.mean() + half) if half is not None else None,
                    "wins": int((delta > 1e-14).sum()),
                    "ties": int((np.abs(delta) <= 1e-14).sum()),
                    "losses": int((delta < -1e-14).sum()),
                    "wilcoxon_p": float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0,
                    "standardized_paired_effect": float(delta.mean() / sd) if sd else None,
                })
    return pd.DataFrame(rows)
