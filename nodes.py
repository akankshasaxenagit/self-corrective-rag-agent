import json
import re
from langchain_core.prompts import ChatPromptTemplate
from graph_state import GraphState
from chains import retriever, llm, prompt as rag_prompt

def _extract_json(text: str) -> dict:
    """Strip markdown code fences and parse JSON, with a safe fallback."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return {}


def route_question(state: GraphState) -> str:
    router_prompt = ChatPromptTemplate.from_template(
        """Classify the user's message as exactly one word: "retrieve" if it is a real question that needs looking up in documents, or "casual" if it is just a greeting or small talk.

Message: {question}

Respond with only one word: retrieve or casual"""
    )
    chain = router_prompt | llm
    result = chain.invoke({"question": state["question"]})
    answer = result.content.strip().lower()
    if "casual" in answer:
        return "casual"
    return "retrieve"


def retrieve_documents(state: GraphState) -> GraphState:
    docs = retriever.invoke(state["question"])
    state["documents"] = [doc.page_content for doc in docs]
    return state


def grade_documents(state: GraphState) -> GraphState:
    grade_prompt = ChatPromptTemplate.from_template(
        """You are grading whether retrieved documents are relevant to a question.

Question: {question}

Documents:
{documents}

Respond with ONLY strict JSON, no other text: {{"relevant": true}} or {{"relevant": false}}"""
    )
    chain = grade_prompt | llm
    result = chain.invoke({
        "question": state["question"],
        "documents": "\n\n".join(state["documents"]),
    })
    parsed = _extract_json(result.content)
    state["is_relevant"] = parsed.get("relevant", False)
    return state


def rewrite_query(state: GraphState) -> GraphState:
    rewrite_prompt = ChatPromptTemplate.from_template(
        """The following question did not retrieve relevant documents. Rewrite it to be more specific or use different terminology that might match the source material better. Return ONLY the rewritten question, nothing else.

Original question: {question}"""
    )
    chain = rewrite_prompt | llm
    result = chain.invoke({"question": state["question"]})
    state["question"] = result.content.strip()
    state["retry_count"] = state.get("retry_count", 0) + 1
    return state


def generate_answer(state: GraphState) -> GraphState:
    context = "\n\n".join(state["documents"])
    chain = rag_prompt | llm
    result = chain.invoke({"context": context, "question": state["question"]})
    state["generation"] = result.content
    return state


def check_hallucination(state: GraphState) -> GraphState:
    check_prompt = ChatPromptTemplate.from_template(
        """Check whether every claim in the answer below is directly supported by the documents. Respond with ONLY strict JSON, no other text: {{"grounded": true}} or {{"grounded": false}}

Documents:
{documents}

Answer:
{answer}"""
    )
    chain = check_prompt | llm

    def _check(answer_text: str) -> bool:
        result = chain.invoke({
            "documents": "\n\n".join(state["documents"]),
            "answer": answer_text,
        })
        parsed = _extract_json(result.content)
        return parsed.get("grounded", True)

    state["is_grounded"] = _check(state["generation"])

    if not state["is_grounded"]:
        context = "\n\n".join(state["documents"])
        strict_prompt = ChatPromptTemplate.from_template(
            """Answer using ONLY information explicitly stated in the documents below. Do not add anything not directly supported. If the documents contradict a common assumption, follow the documents exactly.

Documents:
{context}

Question: {question}

Answer:"""
        )
        chain2 = strict_prompt | llm
        result2 = chain2.invoke({"context": context, "question": state["question"]})
        regenerated = result2.content

        # Verify the regenerated answer too, instead of trusting it blindly
        still_grounded = _check(regenerated)
        if still_grounded:
            state["generation"] = regenerated
            state["is_grounded"] = True
        else:
            state["generation"] = (
                regenerated
                + "\n\n⚠️ Note: this answer could not be fully verified against the source "
                "documents after a retry. Please double-check it against your notes."
            )
            state["is_grounded"] = False

    return state