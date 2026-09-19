from langgraph.graph import StateGraph, END
from graph_state import GraphState
from nodes import (
    route_question,
    retrieve_documents,
    grade_documents,
    rewrite_query,
    generate_answer,
    check_hallucination,
)

MAX_RETRIES = 2

def decide_after_router(state: GraphState) -> str:
    return route_question(state)

def decide_after_grade(state: GraphState) -> str:
    if state["is_relevant"] or state.get("retry_count", 0) >= MAX_RETRIES:
        return "generate"
    return "rewrite"

def casual_response(state: GraphState) -> GraphState:
    state["generation"] = "Hi! Ask me a question about the documents I have access to."
    return state


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("grade", grade_documents)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("generate", generate_answer)
workflow.add_node("check_hallucination", check_hallucination)
workflow.add_node("casual", casual_response)

workflow.set_conditional_entry_point(
    decide_after_router,
    {
        "retrieve": "retrieve",
        "casual": "casual",
    },
)

workflow.add_edge("retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    decide_after_grade,
    {
        "generate": "generate",
        "rewrite": "rewrite",
    },
)

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", "check_hallucination")
workflow.add_edge("check_hallucination", END)
workflow.add_edge("casual", END)

app = workflow.compile()


def run_agent(question: str) -> str:
    initial_state = {
        "question": question,
        "documents": [],
        "generation": "",
        "retry_count": 0,
        "is_relevant": False,
        "is_grounded": True,
    }
    final_state = app.invoke(initial_state)
    print(f"\n[DEBUG] retry_count={final_state.get('retry_count')}, is_relevant={final_state.get('is_relevant')}, is_grounded={final_state.get('is_grounded')}")
    return final_state["generation"]


if __name__ == "__main__":
    print("Ask a question (type 'quit' to exit):")
    while True:
        question = input("\n> ")
        if question.lower() == "quit":
            break
        answer = run_agent(question)
        print(f"\nAnswer: {answer}")