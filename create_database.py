
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Load PDF
pdf_path = Path("document loaders/deeplearning.pdf")
docs = PyPDFLoader(str(pdf_path)).load()
print(f"Loaded {len(docs)} pages")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(docs)
chunks = [c for c in chunks if c.page_content.strip()]

print(f"Created {len(chunks)} chunks")

# Initialize embeddings
embedding_model = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://127.0.0.1:11434"
)

# Create empty Chroma collection
vectorstore = Chroma(
    collection_name="pdf_debug",
    embedding_function=embedding_model,
    persist_directory="./chroma_debug"
)

# Add chunks one at a time to identify the failure
for i, chunk in enumerate(chunks):
    try:
        print(
            f"Processing chunk {i + 1}/{len(chunks)} "
            f"({len(chunk.page_content)} characters)"
        )

        vectorstore.add_documents([chunk])

    except Exception as e:
        print(f"\nFAILED at chunk {i + 1}")
        print(f"Page metadata: {chunk.metadata}")
        print(f"Error: {type(e).__name__}: {e}")
        raise

print("\nAll chunks stored successfully!")
