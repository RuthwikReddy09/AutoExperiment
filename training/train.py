from typing import Dict, Any

import mlflow
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from training.data import build_loaders
from training.model import build_model
from tracking.mlflow_utils import flatten_dict


def _accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_model(config: Dict[str, Any], run_name: str = "train") -> Dict[str, float]:
    device = torch.device(config["device"])
    train_loader, val_loader = build_loaders(config)
    model = build_model(config).to(device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer_name = config.get("optimizer", "adamw").lower()
    if optimizer_name == "sgd":
        optimizer = SGD(model.parameters(), lr=float(config["lr"]), momentum=0.9, weight_decay=float(config["weight_decay"]))
    else:
        optimizer = AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))

    epochs = int(config["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    gradient_clip = config.get("gradient_clip", None)
    nan_detected = False

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(flatten_dict(config))

        for epoch in range(epochs):
            model.train()
            train_loss_sum = 0.0
            train_acc_sum = 0.0
            n_train = 0

            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y)

                if torch.isnan(loss) or torch.isinf(loss):
                    nan_detected = True
                    break

                loss.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
                optimizer.step()

                bs = y.size(0)
                train_loss_sum += loss.item() * bs
                train_acc_sum += _accuracy(logits.detach(), y) * bs
                n_train += bs

            scheduler.step()
            if nan_detected:
                break

            model.eval()
            val_loss_sum = 0.0
            val_acc_sum = 0.0
            n_val = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    logits = model(x)
                    loss = loss_fn(logits, y)
                    bs = y.size(0)
                    val_loss_sum += loss.item() * bs
                    val_acc_sum += _accuracy(logits, y) * bs
                    n_val += bs

            epoch_metrics = {
                "train_loss": train_loss_sum / max(n_train, 1),
                "train_accuracy": train_acc_sum / max(n_train, 1),
                "val_loss": val_loss_sum / max(n_val, 1),
                "val_accuracy": val_acc_sum / max(n_val, 1),
            }
            mlflow.log_metrics(epoch_metrics, step=epoch)

        metrics = dict(epoch_metrics) if not nan_detected else {
            "train_loss": 999.0,
            "train_accuracy": 0.0,
            "val_loss": 999.0,
            "val_accuracy": 0.0,
        }
        metrics["nan_detected"] = 1.0 if nan_detected else 0.0
        mlflow.log_metrics({f"final_{k}": v for k, v in metrics.items()})
        return metrics
