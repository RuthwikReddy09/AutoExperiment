from langgraph.graph import StateGraph, END

from state import ExperimentState
from agents.observer import observer_node
from agents.planner import planner_node
from agents.executor import executor_node
from agents.evaluator import evaluator_node


def should_continue(state):
    return "stop" if state["stop"] else "continue"


def build_graph():
    graph = StateGraph(ExperimentState)
    graph.add_node("observer", observer_node)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("evaluator", evaluator_node)
    graph.set_entry_point("observer")
    graph.add_edge("observer", "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "evaluator")
    graph.add_conditional_edges("evaluator", should_continue, {"continue": "observer", "stop": END})
    return graph.compile()
