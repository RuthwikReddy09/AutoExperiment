import mlflow


def setup_mlflow(experiment_name: str):
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(experiment_name)


def flatten_dict(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out
