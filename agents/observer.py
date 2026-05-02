def observer_node(state):
    metrics = state.get("metrics", {})
    issue = None

    train_acc = metrics.get("train_accuracy")
    val_acc = metrics.get("val_accuracy")
    train_loss = metrics.get("train_loss")
    val_loss = metrics.get("val_loss")

    if metrics.get("nan_detected", 0.0) == 1.0:
        issue = "nan_failure"
    elif train_acc is not None and val_acc is not None and train_acc - val_acc > 0.18:
        issue = "overfitting"
    elif train_acc is not None and val_acc is not None and train_acc < 0.45 and val_acc < 0.45:
        issue = "underfitting"
    elif train_loss is not None and val_loss is not None and val_loss > train_loss * 1.8:
        issue = "generalization_gap"
    else:
        issue = "tune_hyperparameters"

    print(f"[Observer] issue={issue}, metrics={metrics}")
    return {**state, "issue": issue}
