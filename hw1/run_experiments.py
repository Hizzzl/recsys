"""Common experiment runner; run from experiments.ipynb or directly."""

from __future__ import annotations

import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

import config
from evaluation import evaluate
from models.item_knn import ItemKNN
from models.ials import ImplicitALS
from models.ease import EASE
from models.custom_als import CustomTargetALS
from models.popularity import Popularity

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def make_model(name, parameters):
    if name in {"MostPopular", "RecentPopular"}:
        return Popularity(**parameters)
    if name == "FM":
        from models.fm import FactorizationMachine
        return FactorizationMachine(**parameters, seed=config.SEED)
    cls = {"ItemKNN": ItemKNN, "iALS": ImplicitALS, "EASE": EASE, "CustomALS": CustomTargetALS}[name]
    extra = {"seed": config.SEED} if name in {"iALS", "CustomALS"} else {}
    return cls(**parameters, **extra)


def _measure(name, parameters, history, heldout, users, data, cutoff):
    model = make_model(name, parameters)
    started = time.perf_counter()
    model.fit(history, data["n_users"], data["n_items"], cutoff)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    metrics = evaluate(model, history, heldout, users, data["n_users"], data["n_items"])
    score_seconds = time.perf_counter() - started
    record = {"model": name, "parameters": parameters,
              "NDCG@10": metrics["NDCG@10"], "Recall@10": metrics["Recall@10"],
              "fit_seconds": fit_seconds, "evaluation_seconds": score_seconds,
              "users": metrics["users"]}
    return model, metrics, record


def _save(name, value):
    (RESULTS / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def run_all(data):
    """Tune on validation, freeze every winner, then evaluate held-out test."""
    RESULTS.mkdir(exist_ok=True)
    _save("data_summary.json", data["summary"])
    _save("experiment_config.json", {
        "seed": config.SEED, "primary_metric": config.PRIMARY_METRIC,
        "search_spaces": config.SEARCH_SPACES, "relevant_rating": config.RELEVANT_RATING,
        "max_evaluation_users": config.MAX_EVALUATION_USERS,
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "pandas": pd.__version__,
        "blas_threads": 1, "platform": platform.platform(),
    })
    trials, winners = [], {}
    with threadpool_limits(limits=1):
        for name, space in config.SEARCH_SPACES.items():
            best_score = -np.inf
            for index, parameters in enumerate(space):
                model, metrics, record = _measure(name, parameters, data["train"], data["validation"],
                    data["validation_users"], data, data["train_cutoff_ms"])
                record["trial"] = index
                trials.append(record)
                print(f"validation {name} {index + 1}/{len(space)}: NDCG={record['NDCG@10']:.5f}, Recall={record['Recall@10']:.5f}, fit={record['fit_seconds']:.1f}s", flush=True)
                if record[config.PRIMARY_METRIC] > best_score:
                    best_score = record[config.PRIMARY_METRIC]
                    winners[name] = dict(parameters)
                del model, metrics
                gc.collect()
                _save("validation_trials.json", trials)
        # Freeze all choices before the first test score is computed.
        _save("selected_parameters.json", winners)
        validation_table = pd.DataFrame([{**r, "parameters": json.dumps(r["parameters"], sort_keys=True)} for r in trials])
        validation_table.to_csv(RESULTS / "validation_trials.csv", index=False)
        history = pd.concat([data["train"], data["validation"]], ignore_index=True)
        final, details, sample = [], {}, {}
        for name, parameters in winners.items():
            model, metrics, record = _measure(name, parameters, history, data["test"],
                data["test_users"], data, data["validation_cutoff_ms"])
            final.append(record)
            details[name] = {key: metrics[key] for key in ["per_user_ndcg", "per_user_recall"]}
            sample[name] = metrics["recommendations"][:3]
            print(f"test {name}: NDCG={record['NDCG@10']:.5f}, Recall={record['Recall@10']:.5f}, fit={record['fit_seconds']:.1f}s", flush=True)
            del model, metrics
            gc.collect()
            _save("test_results.json", final)
        # Predeclared controlled comparisons; test is not used to alter any choice.
        controlled_params = {**winners["iALS"], "half_life_days": 365.0}
        model, controlled_metrics, controlled_record = _measure("CustomALS", controlled_params,
            history, data["test"], data["test_users"], data, data["validation_cutoff_ms"])
        controlled_record["comparison"] = "same factors/regularization/alpha/iterations as selected iALS"
        del model
        gc.collect()
        # Fixed post-hoc diagnostics; validation selections remain unchanged.
        diagnostics = []
        for name, parameters in [("MostPopular", {}), ("RecentPopular", {"half_life_days": 365.0})]:
            model, metrics, record = _measure(name, parameters, history, data["test"],
                data["test_users"], data, data["validation_cutoff_ms"])
            record["purpose"] = "post-hoc non-personalized diagnostic; no tuning"
            diagnostics.append(record)
            details[name] = {key: metrics[key] for key in ["per_user_ndcg", "per_user_recall"]}
            print(f"diagnostic {name}: NDCG={record['NDCG@10']:.5f}, Recall={record['Recall@10']:.5f}", flush=True)
            del model, metrics
        _save("popularity_diagnostics.json", diagnostics)
        ids_only_params = {**winners["FM"], "use_metadata": False}
        model, ids_metrics, ids_record = _measure("FM", ids_only_params,
            history, data["test"], data["test_users"], data, data["validation_cutoff_ms"])
        ids_record["comparison"] = "same selected FM settings, ID features only"
        del model
        gc.collect()
    _save("controlled_comparisons.json", [controlled_record, ids_record])
    details["CustomALS_controlled"] = {key: controlled_metrics[key] for key in ["per_user_ndcg", "per_user_recall"]}
    details["FM_ID_only"] = {key: ids_metrics[key] for key in ["per_user_ndcg", "per_user_recall"]}
    _save("per_user_test_metrics.json", {"users": data["test_users"].tolist(), "metrics": details})
    _save("recommendation_examples.json", {"users": data["test_users"][:3].tolist(), "recommendations": sample})
    table = pd.DataFrame([{**r, "parameters": json.dumps(r["parameters"], sort_keys=True)} for r in final])
    table = table.sort_values(config.PRIMARY_METRIC, ascending=False).reset_index(drop=True)
    table.to_csv(RESULTS / "test_results.csv", index=False)
    return {"validation": validation_table, "test": table, "winners": winners,
            "comparisons": pd.DataFrame([controlled_record, ids_record]),
            "diagnostics": pd.DataFrame(diagnostics), "per_user": details}


if __name__ == "__main__":
    from data_utils import prepare_data
    dataset = prepare_data()
    print(json.dumps(dataset["summary"], indent=2), flush=True)
    result = run_all(dataset)
    print(result["test"].to_string(index=False))
