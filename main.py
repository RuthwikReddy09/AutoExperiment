import json

import mlflow

from graph import build_graph
from tracking.mlflow_utils import setup_mlflow
from utils.config import parse_args, load_yaml, seed_everything, resolve_device, apply_cli_overrides


def build_initial_state(cfg):
    exp = cfg["experiment"]
    train = cfg["training"].copy()
    agent = cfg.get("agent", {})
    optuna_cfg = cfg.get("optuna", {})

    train["seed"] = exp.get("seed", 42)
    train["device"] = resolve_device(train.get("device", "auto"))
    train["metric_to_optimize"] = exp.get("metric_to_optimize", "val_accuracy")
    train["direction"] = exp.get("direction", "maximize")
    train["max_iterations"] = exp.get("max_iterations", 3)
    train["use_llm_planner"] = agent.get("use_llm_planner", False)
    train["openai_model"] = agent.get("openai_model", "gpt-4o-mini")
    train["optuna"] = optuna_cfg

    initial_best = -1e9 if train["direction"] == "maximize" else 1e9
    return {
        "experiment_name": exp["name"],
        "iteration": 0,
        "config": train,
        "search_space": cfg.get("search_space", {}),
        "metrics": {},
        "issue": None,
        "plan": None,
        "best_score": initial_best,
        "best_config": {},
        "history": [],
        "stop": False,
    }


def main():
    args = parse_args()
    cfg = apply_cli_overrides(load_yaml(args.config), args)
    seed_everything(int(cfg["experiment"].get("seed", 42)))
    setup_mlflow(cfg["experiment"]["name"])

    app = build_graph()
    state = build_initial_state(cfg)

    with mlflow.start_run(run_name="autoexperiment_controller"):
        mlflow.log_param("use_llm_planner", state["config"]["use_llm_planner"])
        mlflow.log_param("device", state["config"]["device"])
        final_state = app.invoke(state)
        mlflow.log_metric("best_score", final_state["best_score"])
        mlflow.log_text(json.dumps(final_state["best_config"], indent=2), "best_config.json")
        mlflow.log_text(json.dumps(final_state["history"], indent=2, default=str), "history.json")

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")
    print(f"Best score: {final_state['best_score']:.4f}")
    print("Best config:")
    print(json.dumps(final_state["best_config"], indent=2, default=str))


if __name__ == "__main__":
    main()
