from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PERSIST_DIR = "./chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below. If the context doesn't contain the answer, say you don't have enough information to answer.

Context:
{context}

Question: {question}

Answer:"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def answer_question(question: str) -> str:
    return rag_chain.invoke(question)

if __name__ == "__main__":
    print("Ask a question (type 'quit' to exit):")
    while True:
        question = input("\n> ")
        if question.lower() == "quit":
            break
        answer = answer_question(question)
        print(f"\nAnswer: {answer}")