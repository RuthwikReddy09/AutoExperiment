from typing import TypedDict, Optional, Dict, Any, List

class ExperimentState(TypedDict):
    experiment_name: str
    iteration: int
    config: Dict[str, Any]
    search_space: Dict[str, Any]
    metrics: Dict[str, float]
    issue: Optional[str]
    plan: Optional[Dict[str, Any]]
    best_score: float
    best_config: Dict[str, Any]
    history: List[Dict[str, Any]]
    stop: bool
