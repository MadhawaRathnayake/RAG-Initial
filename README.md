# RAG For Beginners

Small, progressive examples for building a Retrieval-Augmented Generation (RAG) application with Python, LangChain, OpenAI models, and ChromaDB.

The project demonstrates how to:

- Load local text documents.
- Split documents into searchable chunks.
- Create embeddings and persist them in ChromaDB.
- Retrieve relevant context for a question.
- Generate answers grounded in the retrieved documents.
- Rewrite follow-up questions using conversation history.
- Compare character, recursive, semantic, and title-based chunking.
- Process multimodal PDF content, including tables and images, in the notebook.

## Project Structure

| File | Description |
| --- | --- |
| `1_ingestion_pipeline.py` | Loads `.txt` files from `docs/`, chunks them, and creates `db/chroma.db`. |
| `2_retrieval_pipeline.py` | Retrieves the most relevant chunks for a query. |
| `3_response_generator.py` | Uses retrieved chunks to generate an answer with an OpenAI chat model. |
| `4_history_aware_generation.py` | Adds conversational question rewriting and chat history. |
| `5_character_text_splitter.py` | Compares basic and recursive character splitting. |
| `6_semantic_chunking.py` | Demonstrates embedding-based semantic chunking. |
| `multi_model_rag.ipynb` | Multimodal PDF ingestion, table/image handling, summarisation, retrieval, and answer generation. |
| `docs/` | Source documents used by the text ingestion pipeline. |

## Requirements

- Python 3.10 or newer
- An OpenAI API key
- Windows, macOS, or Linux
- Tesseract OCR and Poppler for the multimodal PDF notebook

## Setup on Windows

Create and activate a virtual environment from the project directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the packages used by the script examples:

```powershell
python -m pip install langchain langchain-community langchain-openai langchain-chroma langchain-text-splitters langchain-experimental python-dotenv
```

For the multimodal notebook, also install:

```powershell
python -m pip install "unstructured[all-docs]"
```

Install the external PDF/OCR tools with `winget` in an Administrator PowerShell if required:

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
winget install --id oschwartz10612.Poppler -e
```

Restart VS Code after installation so the notebook kernel receives the updated `PATH`. If Tesseract is installed but is not visible to the kernel, add this before calling `partition_pdf`:

```python
import os

os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"
```

## Environment Variables

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

The `.env` file is ignored by Git. Never commit a real API key.

## Run the Text RAG Pipeline

Activate the virtual environment, then run the scripts in order:

```powershell
.\venv\Scripts\Activate.ps1
python 1_ingestion_pipeline.py
python 2_retrieval_pipeline.py
python 3_response_generator.py
python 4_history_aware_generation.py
```

The ingestion step reads `.txt` files from `docs/` and creates a local Chroma database under `db/`. Run ingestion again when the source documents change.

Scripts 5 and 6 are standalone chunking demonstrations:

```powershell
python 5_character_text_splitter.py
python 6_semantic_chunking.py
```

## Run the Notebook

Open `multi_model_rag.ipynb` in VS Code, select the interpreter at `venv\Scripts\python.exe`, and run the cells from top to bottom.

The notebook currently expects this PDF path:

```text
docs/attention-is-all-you-need.pdf
```

Add that PDF or change `file_path` in the notebook before running the partitioning cell. The notebook may create local Chroma databases and JSON exports; these are intentionally ignored by Git.

## RAG Flow

```text
Source documents
      |
      v
Load and split text ------> Create embeddings ------> ChromaDB
                                                        |
User question ---> Retrieve relevant chunks -----------+
                                                        |
                                                        v
                                              OpenAI response
```

## Notes

- The examples use OpenAI embeddings and chat models, so API usage may incur charges.
- The scripts use `text-embedding-3-small` and `gpt-4o`.
- The generated `db/`, `dbv1/`, `dbv2/`, JSON exports, virtual environment, and secrets are excluded by `.gitignore`.