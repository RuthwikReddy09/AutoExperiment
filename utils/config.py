import argparse
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="AutoExperiment LangGraph runner")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--use-llm-planner", action="store_true")
    parser.add_argument("--full-run", action="store_true", help="Disable quick_debug subset training")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def apply_cli_overrides(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    cfg = cfg.copy()
    cfg["agent"] = cfg.get("agent", {}).copy()
    cfg["training"] = cfg.get("training", {}).copy()
    if args.use_llm_planner:
        cfg["agent"]["use_llm_planner"] = True
    if args.full_run:
        cfg["training"]["quick_debug"] = False
    return cfg
