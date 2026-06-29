import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="centered"
)

# ── Constants ──────────────────────────────────────────────────
CORPUS_PATH = "./hr_docs/"
REFUSAL_MESSAGE = (
    "I'm sorry, I can only answer HR-related questions based on "
    "Zyro Dynamics policy documents. Please consult the appropriate "
    "resource for this query."
)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an expert HR assistant for Zyro Dynamics Pvt. Ltd.
Your job is to answer employee questions STRICTLY based on the provided HR policy documents.

RULES:
1. Answer ONLY using the context provided below — never use outside knowledge.
2. If the context contains the answer, give a clear, detailed, and professional response.
3. If the question is NOT related to HR policies, company rules, leave, benefits, conduct,
   performance, travel, IT, onboarding, separation, or any topic covered in Zyro Dynamics
   HR documents — respond with EXACTLY the refusal message.
4. Never make up information, numbers, or policies not found in the context.
5. Always be professional, concise, and helpful.
6. If the answer is partially available, share what is available and mention the limitation.

CONTEXT FROM HR POLICY DOCUMENTS:
{context}

EMPLOYEE QUESTION:
{question}

ANSWER:
""")

OOS_PROMPT = ChatPromptTemplate.from_template("""
You are a strict classifier. Your ONLY job is to decide if a question is related to HR policies or not.

HR-related topics include:
- Leave policies (EL, SL, maternity, paternity, etc.)
- Work from home / hybrid / remote work
- Code of conduct and ethics
- Performance reviews, appraisals, PIP
- Compensation, salary, CTC, benefits, grades
- IT and data security policies
- Prevention of sexual harassment (POSH)
- Onboarding, probation, separation, full & final settlement
- Travel and expense reimbursements
- Company profile, culture, employee handbook
- Any Zyro Dynamics internal policy or workplace guideline

Respond with ONLY one word — either YES or NO.
YES = the question is HR-related
NO = the question is NOT HR-related

Question: {question}

Answer (YES or NO only):
""")

# ── Pipeline loader (cached) ───────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading HR policy documents...")
def load_pipeline():
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
    )

    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.6}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
        api_key=st.secrets["GROQ_API_KEY"]
    )

    return retriever, llm

def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        filename = os.path.basename(source)
        formatted.append(
            f"[Source {i+1}: {filename} | Page {page}]\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(formatted)

def ask_bot(question, retriever, llm):
    try:
        # Layer 1: Out-of-scope guard
        oos_prompt = OOS_PROMPT.invoke({"question": question})
        oos_response = llm.invoke(oos_prompt)
        classification = StrOutputParser().invoke(oos_response).strip().upper()

        if "NO" in classification:
            return REFUSAL_MESSAGE, []

        # Layer 2: RAG pipeline
        docs = retriever.invoke(question)
        context = format_docs(docs)
        prompt = RAG_PROMPT.invoke({"context": context, "question": question})
        response = llm.invoke(prompt)
        answer = StrOutputParser().invoke(response).strip()

        if not answer or len(answer) < 10:
            return REFUSAL_MESSAGE, []

        return answer, docs

    except Exception as e:
        return f"An error occurred: {str(e)}", []

# ── UI ─────────────────────────────────────────────────────────
st.title("🏢 Zyro Dynamics HR Help Desk")
st.markdown(
    "Ask any HR-related question about Zyro Dynamics policies. "
    "I'll answer based on official company documents."
)
st.divider()

# Load pipeline
retriever, llm = load_pipeline()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Source Documents"):
                for src in msg["sources"]:
                    filename = os.path.basename(src.metadata.get("source", "Unknown"))
                    page = src.metadata.get("page", "N/A")
                    st.markdown(f"**{filename}** — Page {page}")
                    st.caption(src.page_content[:300] + "...")

# Chat input
if question := st.chat_input("Ask your HR question here..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and show answer
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            answer, sources = ask_bot(question, retriever, llm)
        st.markdown(answer)
        if sources:
            with st.expander("📄 Source Documents"):
                for src in sources:
                    filename = os.path.basename(src.metadata.get("source", "Unknown"))
                    page = src.metadata.get("page", "N/A")
                    st.markdown(f"**{filename}** — Page {page}")
                    st.caption(src.page_content[:300] + "...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })