def evaluator_node(state):
    metric_name = state["config"].get("metric_to_optimize", "val_accuracy")
    direction = state["config"].get("direction", "maximize")
    current_score = float(state["metrics"].get(metric_name, 0.0))
    best_score = float(state["best_score"])
    improved = current_score > best_score if direction == "maximize" else current_score < best_score

    best_config = state["best_config"]
    if improved:
        best_score = current_score
        best_config = state["config"].copy()
        print(f"[Evaluator] New best {metric_name}: {best_score:.4f}")
    else:
        print(f"[Evaluator] No improvement. Best {metric_name}: {best_score:.4f}")

    history = state["history"] + [{
        "iteration": state["iteration"],
        "issue": state["issue"],
        "plan": state["plan"],
        "metrics": state["metrics"],
        "config": state["config"],
    }]
    stop = state["iteration"] >= int(state["config"].get("max_iterations", 3))
    return {**state, "best_score": best_score, "best_config": best_config, "history": history, "stop": stop}
