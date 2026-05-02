from typing import Dict, Any

import mlflow
import optuna

from training.train import train_model


def suggest_from_space(trial: optuna.Trial, base_config: Dict[str, Any], search_space: Dict[str, Any]) -> Dict[str, Any]:
    cfg = base_config.copy()
    for name, spec in search_space.items():
        kind = spec.get("type")
        if kind == "float":
            cfg[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
        elif kind == "int":
            cfg[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log", False)))
        elif kind == "categorical":
            cfg[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported search-space type for {name}: {kind}")
    return cfg


def executor_node(state):
    base_config = state["config"].copy()
    search_space = state["search_space"]
    optuna_cfg = base_config.get("optuna", {})
    n_trials = int(optuna_cfg.get("n_trials", 4))
    timeout = optuna_cfg.get("timeout_seconds", None)
    timeout = None if timeout in (None, "null") else int(timeout)

    print(f"[Executor] iteration={state['iteration']} optuna_trials={n_trials}")

    def objective(trial: optuna.Trial):
        cfg = suggest_from_space(trial, base_config, search_space)
        metrics = train_model(cfg, run_name=f"iter_{state['iteration']}_optuna_trial_{trial.number}")
        trial.set_user_attr("metrics", metrics)
        return metrics[base_config.get("metric_to_optimize", "val_accuracy")]

    direction = base_config.get("direction", "maximize")
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_config = base_config.copy()
    best_config.update(study.best_params)
    metrics = train_model(best_config, run_name=f"iter_{state['iteration']}_best_retrain")

    mlflow.log_metric(f"iteration_{state['iteration']}_best_val_accuracy", metrics.get("val_accuracy", 0.0))

    print(f"[Executor] val_accuracy={metrics.get('val_accuracy', 0.0):.4f}")
    return {**state, "config": best_config, "metrics": metrics, "iteration": state["iteration"] + 1}
