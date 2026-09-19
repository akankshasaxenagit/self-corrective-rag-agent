import streamlit as st
from graph import run_agent

st.set_page_config(page_title="Self-Corrective RAG Agent", page_icon="🔎")
st.title("🔎 Self-Corrective RAG Agent")
st.caption("Ask a question about the DBMS documents loaded into this agent.")

if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

question = st.chat_input("Ask a question...")

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = run_agent(question)
        st.markdown(answer)

    st.session_state.history.append(("assistant", answer))