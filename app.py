import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langsmith import traceable

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

# ── Environment ──────────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = "zyro-rag-challenge"
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

CORPUS_PATH = "./"   # folder containing all 11 HR PDFs

REFUSAL_MESSAGE = (
    "I'm sorry, I can only answer HR-related questions based on "
    "Zyro Dynamics policy documents. Your question appears to be outside "
    "that scope. Please reach out to hr.helpdesk@zyrodyanmics.com for other queries."
)

# ── Prompts ──────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an HR assistant for Zyro Dynamics. Answer employee questions \
accurately using ONLY the context provided below. Be concise and factual.

If the answer is not in the context, say: 'I could not find this information in the Zyro Dynamics HR policy documents.'

Always cite the source document at the end of your answer.

Context:
{context}"""),
    ("human", "{question}")
])

OOS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a classifier. Determine if the question is related to HR, \
employment, workplace policies, leave, salary, benefits, performance, travel expenses, \
IT security, onboarding, separation, or company conduct.
Reply with ONLY one word: YES if it is HR-related, NO if it is not."""),
    ("human", "{question}")
])

# ── Pipeline (cached so it only loads once) ───────────────────────
@st.cache_resource(show_spinner="Loading HR policy documents…")
def build_pipeline():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10}
    )

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512)
    return retriever, llm


def format_docs(docs):
    parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown").split("/")[-1]
        parts.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


@traceable(name="zyro-ask-bot")
def ask_bot(question: str, retriever, llm) -> dict:
    # Guardrail: classify question first
    cls = llm.invoke(OOS_PROMPT.invoke({"question": question})).content.strip().upper()
    if "NO" in cls:
        return {"answer": REFUSAL_MESSAGE, "sources": []}

    # RAG
    docs = retriever.invoke(question)
    context = format_docs(docs)
    answer = llm.invoke(RAG_PROMPT.invoke({"context": context, "question": question})).content
    sources = list({doc.metadata.get("source", "").split("/")[-1] for doc in docs})
    return {"answer": answer, "sources": sources}


# ── UI ────────────────────────────────────────────────────────────
st.title("🏢 Zyro Dynamics HR Help Desk")
st.caption("Ask any HR policy question. Powered by RAG over 11 internal policy documents.")
st.divider()

retriever, llm = build_pipeline()

# Suggested questions sidebar
with st.sidebar:
    st.header("💡 Try asking…")
    suggestions = [
        "How many days of earned leave do I get per year?",
        "What is the notice period for an L4 employee?",
        "Can I work from home as an L2 employee?",
        "How do I file a POSH complaint?",
        "What is the hotel limit for L6 domestic travel?",
        "When is the Annual Performance Review conducted?",
        "What benefits do I get under Group Medical Insurance?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True):
            st.session_state["prefill"] = s

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Source documents"):
                for src in msg["sources"]:
                    st.write(f"• {src}")

# Handle prefilled question from sidebar button
user_input = st.session_state.pop("prefill", None)

# Chat input
typed = st.chat_input("Ask an HR question…")
if typed:
    user_input = typed

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents…"):
            result = ask_bot(user_input, retriever, llm)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("📄 Source documents"):
                for src in result["sources"]:
                    st.write(f"• {src}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
