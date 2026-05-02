import json
import os
from typing import Dict, Any


def heuristic_plan(state) -> Dict[str, Any]:
    issue = state["issue"]
    cfg = state["config"].copy()
    search_space = state["search_space"].copy()

    if issue in {"overfitting", "generalization_gap"}:
        cfg["dropout"] = min(float(cfg.get("dropout", 0.2)) + 0.1, 0.6)
        cfg["weight_decay"] = min(float(cfg.get("weight_decay", 1e-4)) * 2.0, 1e-2)
        rationale = "Increase regularization because validation performance lags training performance."
    elif issue == "underfitting":
        cfg["epochs"] = int(cfg.get("epochs", 3)) + 2
        cfg["lr"] = min(float(cfg.get("lr", 1e-3)) * 1.5, 1e-2)
        rationale = "Increase training budget and learning rate because both train and validation accuracy are low."
    elif issue == "nan_failure":
        cfg["lr"] = max(float(cfg.get("lr", 1e-3)) * 0.1, 1e-6)
        cfg["gradient_clip"] = 1.0
        rationale = "Reduce learning rate and clip gradients to recover from numerical instability."
    else:
        rationale = "Run Optuna around the current configuration to improve validation accuracy."

    return {"config_updates": cfg, "search_space_updates": search_space, "rationale": rationale}


def llm_plan(state) -> Dict[str, Any]:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    model_name = state["config"].get("openai_model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    llm = ChatOpenAI(model=model_name, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an ML experiment planner. Return ONLY valid JSON with keys:
config_updates, search_space_updates, rationale.
Do not add markdown. Keep changes conservative and executable.
"""),
        ("human", """
Current issue: {issue}
Current metrics: {metrics}
Current config: {config}
Current search space: {search_space}
Best score so far: {best_score}
History: {history}

Suggest the next configuration and search-space updates for Optuna.
"""),
    ])
    chain = prompt | llm
    response = chain.invoke({
        "issue": state["issue"],
        "metrics": state["metrics"],
        "config": state["config"],
        "search_space": state["search_space"],
        "best_score": state["best_score"],
        "history": state["history"][-3:],
    })
    try:
        return json.loads(response.content)
    except Exception:
        print("[Planner] LLM returned non-JSON. Falling back to heuristic planner.")
        return heuristic_plan(state)


def planner_node(state):
    use_llm = bool(state["config"].get("use_llm_planner", False))
    has_key = bool(os.getenv("OPENAI_API_KEY"))

    if use_llm and has_key:
        plan = llm_plan(state)
        source = "llm"
    else:
        plan = heuristic_plan(state)
        source = "heuristic"

    new_config = state["config"].copy()
    new_config.update(plan.get("config_updates", {}))
    new_search_space = state["search_space"].copy()
    new_search_space.update(plan.get("search_space_updates", {}))

    print(f"[Planner:{source}] {plan.get('rationale', '')}")
    return {**state, "config": new_config, "search_space": new_search_space, "plan": {**plan, "source": source}}
