from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(
    page_title="RAG-Based PDF Assistant",
    page_icon=":material/menu_book:",
    layout="wide",
)

APP_DIR = Path(__file__).parent
DATABASE_PATH = APP_DIR / "chroma_db"
EMBEDDING_MODEL = "bge-m3"
CHAT_MODEL = "qwen3:4b"
BATCH_SIZE = 8

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Use ONLY the provided context to answer the question. "
            'If the answer is not present in the context, say: "I could not find the answer in the document."',
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ]
)


@st.cache_resource(show_spinner=False)
def get_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(DATABASE_PATH),
        embedding_function=embeddings,
    )


@st.cache_resource(show_spinner=False)
def get_chat_model() -> ChatOllama:
    return ChatOllama(model=CHAT_MODEL)


def index_pdf(filename: str, pdf_bytes: bytes) -> int:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(pdf_bytes)
        temp_path = Path(temp_file.name)

    try:
        documents = PyPDFLoader(str(temp_path)).load()
    finally:
        temp_path.unlink(missing_ok=True)

    for document in documents:
        document.metadata["source"] = filename

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [chunk for chunk in splitter.split_documents(documents) if chunk.page_content.strip()]
    if not chunks:
        return 0

    vectorstore = get_vectorstore()
    file_id = hashlib.sha256(pdf_bytes).hexdigest()
    progress = st.progress(0, text="Preparing document chunks…")
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        ids = [f"{file_id}-{index}" for index in range(start, start + len(batch))]
        vectorstore.add_documents(batch, ids=ids)
        complete = min(start + len(batch), len(chunks))
        progress.progress(complete / len(chunks), text=f"Indexed {complete:,} of {len(chunks):,} chunks")
    progress.empty()
    return len(chunks)


def source_details(documents):
    return [
        {
            "name": Path(str(document.metadata.get("source", "PDF"))).name,
            "page": int(document.metadata.get("page", 0)) + 1,
            "excerpt": document.page_content[:700],
        }
        for document in documents
    ]


def answer_question(question: str):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    documents = retriever.invoke(question)
    context = "\n\n".join(document.page_content for document in documents)
    response = get_chat_model().invoke(PROMPT.invoke({"context": context, "question": question}))
    return response.content, source_details(documents)


with st.sidebar:
    st.markdown("## :material/menu_book: RAG-Based PDF Assistant")
    st.caption("A focused workspace for exploring your PDFs.")
    st.subheader("Add a document")
    uploaded_file = st.file_uploader(
        "Choose a PDF book",
        type=["pdf"],
        help="The PDF is processed locally and stored in your Chroma database.",
    )

    if uploaded_file:
        st.caption(f"Selected: **{uploaded_file.name}** · {uploaded_file.size / (1024 * 1024):.1f} MB")

    if st.button(
        "Create vector database",
        type="primary",
        icon=":material/database:",
        width="stretch",
        disabled=uploaded_file is None,
    ):
        try:
            with st.status("Building your searchable book…", expanded=True) as status:
                chunk_count = index_pdf(uploaded_file.name, uploaded_file.getvalue())
                if chunk_count:
                    status.update(label="Your book is ready", state="complete", expanded=False)
                    st.success(f"Indexed {chunk_count:,} text chunks from **{uploaded_file.name}**.")
                else:
                    status.update(label="No readable text found", state="error", expanded=True)
                    st.warning("This PDF did not contain extractable text.")
        except Exception as exc:
            st.error(f"Could not create the vector database: {exc}")

    st.divider()
    st.subheader("Workspace")
    if DATABASE_PATH.exists():
        try:
            chunk_total = get_vectorstore()._collection.count()
        except Exception:
            chunk_total = 0
    else:
        chunk_total = 0

    if chunk_total:
        st.badge("Document indexed", icon=":material/check_circle:", color="green")
        st.metric("Searchable chunks", f"{chunk_total:,}")
    else:
        st.badge("Ready for a PDF", icon=":material/upload_file:", color="gray")
    st.caption(f"Embeddings · `{EMBEDDING_MODEL}`")
    st.caption(f"Answer model · `{CHAT_MODEL}`")


st.markdown(":green-badge[LOCAL DOCUMENT WORKSPACE]")
st.title("Your PDF" \
", ready to explore.")
st.write("Ask focused questions and get answers grounded in the passages retrieved from your PDF.")

if chunk_total:
    metric_col, help_col = st.columns([1, 3], vertical_alignment="center")
    with metric_col:
        with st.container(border=True):
            st.metric("Indexed passages", f"{chunk_total:,}")
    with help_col:
        with st.container(border=True):
            st.markdown("**Ask about a concept, chapter, or detail**")
            st.caption("Answers use the indexed PDF and include source page references when available.")

    st.subheader("Ask your book")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("View retrieved passages"):
                    for source in message["sources"]:
                        st.markdown(f"**{source['name']} · page {source['page']}**")
                        st.caption(source["excerpt"])

    question = st.chat_input("Ask a question about your book…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your book…"):
                    answer, sources = answer_question(question)
                st.markdown(answer)
                if sources:
                    with st.expander("View retrieved passages"):
                        for source in sources:
                            st.markdown(f"**{source['name']} · page {source['page']}**")
                            st.caption(source["excerpt"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                st.error(f"I couldn't answer that question. Check that Ollama is running. Details: {exc}")
else:
    with st.container(border=True):
        st.markdown("### Start with a PDF")
        st.write("Upload a book in the sidebar and create its vector database. Once indexing finishes, you can ask questions here.")
        st.caption("Supported workflow · PDF upload → local indexing → grounded answers")
