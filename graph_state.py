from typing import TypedDict, List

class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    retry_count: int
    is_relevant: bool
    is_grounded: bool