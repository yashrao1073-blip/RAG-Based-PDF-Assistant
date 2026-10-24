# RAG-Based PDF Assistant

A local Streamlit app that lets you upload a PDF book and ask questions about its contents. It retrieves relevant passages from the document and uses them as context for answers, with source page references.

## Features

- Upload a PDF book through the Streamlit interface.
- Extract and split PDF text into overlapping chunks.
- Embed and store chunks locally with Ollama and Chroma.
- Retrieve relevant passages using maximum marginal relevance (MMR).
- Ask questions in a chat interface and inspect the passages used to answer.
- Process the document locally; the app does not require a hosted LLM API.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/download)
- The `bge-m3` embedding model
- The `qwen3:4b` chat model

Install Ollama, then download the models:

```bash
ollama pull bge-m3
ollama pull qwen3:4b
```

The app uses [BGE-M3](https://ollama.com/library/bge-m3) for embeddings and [Qwen3 4B](https://ollama.com/library/qwen3) for answers.

## Installation

Create and activate a virtual environment from the project directory.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
pip install langchain-ollama
```

`langchain-ollama` is required by the app but is not currently listed in `requirements.txt`.

## Run the app

Make sure Ollama is running, then start Streamlit from the project directory:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal. Upload a PDF in the sidebar and select **Create vector database**. After indexing completes, ask questions in the chat area.

## How it works

1. `PyPDFLoader` extracts text and page metadata from the uploaded PDF.
2. `RecursiveCharacterTextSplitter` creates chunks of 1,000 characters with 200 characters of overlap.
3. Ollama embeds the chunks with `bge-m3`. The app indexes them in batches of eight to keep local requests manageable.
4. Chroma stores the vectors in the local `chroma_db/` directory.
5. For each question, the app retrieves up to four relevant chunks with MMR and passes them to `qwen3:4b` as the answer context.

## Notes

- The Chroma database is generated locally and is excluded from Git by `.gitignore`; re-index a PDF after cloning the repository.
- Text-based PDFs work best. Scanned PDFs without an embedded text layer may not yield readable passages.
- The Ollama models must be available on the same machine as the app.
