from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from models.schemas import State
from steps.impl.router_step import RouterStep
from steps.impl.research_step import ResearchStep
from steps.impl.orchestrator_step import OrchestratorStep
from steps.impl.worker_step import WorkerStep
from steps.impl.reducer_step import MergeContentStep, DecideImagesStep, GenerateAndPlaceImagesStep


def _fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]


def build_graph():
    reducer_graph = StateGraph(State)
    reducer_graph.add_node("merge_content", MergeContentStep.execute)
    reducer_graph.add_node("decide_images", DecideImagesStep.execute)
    reducer_graph.add_node("generate_and_place_images", GenerateAndPlaceImagesStep.execute)
    reducer_graph.add_edge(START, "merge_content")
    reducer_graph.add_edge("merge_content", "decide_images")
    reducer_graph.add_edge("decide_images", "generate_and_place_images")
    reducer_graph.add_edge("generate_and_place_images", END)
    reducer_subgraph = reducer_graph.compile()

    graph = StateGraph(State)
    graph.add_node("router", RouterStep.execute)
    graph.add_node("research", ResearchStep.execute)
    graph.add_node("orchestrator", OrchestratorStep.execute)
    graph.add_node("worker", WorkerStep.execute)
    graph.add_node("reducer", reducer_subgraph)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        RouterStep.route_next,
        {"research": "research", "orchestrator": "orchestrator"},
    )
    graph.add_edge("research", "orchestrator")

    graph.add_conditional_edges("orchestrator", _fanout, ["worker"])
    graph.add_edge("worker", "reducer")
    graph.add_edge("reducer", END)

    return graph.compile()


app = build_graph()
