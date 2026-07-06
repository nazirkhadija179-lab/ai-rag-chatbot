
# AI RAG Chatbot

## Overview

AI RAG Chatbot is a Retrieval-Augmented Generation (RAG) application that answers user questions based on PDF documents.

The project uses LangChain, Hugging Face Embeddings, ChromaDB, Groq LLM, and Streamlit.

---

## Features

- PDF-based Question Answering
- Document Chunking
- Hugging Face Embeddings
- ChromaDB Vector Database
- Semantic Search
- Groq LLM
- Streamlit Interface
- Hugging Face Deployment

---

## Technologies Used

- Python
- Streamlit
- LangChain
- ChromaDB
- Hugging Face Embeddings
- Groq API

---

## Project Structure

```
app.py
build_db.py
requirements.txt
README.md
data/
chroma_db_v2/
```

---

## Workflow

```
PDF Documents
      ↓
Load PDFs
      ↓
Chunking
      ↓
Embeddings
      ↓
ChromaDB
      ↓
User Question
      ↓
Question Embedding
      ↓
Similarity Search
      ↓
Relevant Chunks
      ↓
Groq LLM
      ↓
Final Answer
```

---

## Installation

Install dependencies

```bash
pip install -r requirements.txt
```

Build Vector Database

```bash
python build_db.py
```

Run Application

```bash
streamlit run app.py
```

---

## Deployment

The application is deployed on Hugging Face Spaces.

---

## Note

If the ChromaDB database is missing due to GitHub file size limitations, regenerate it by running:

```bash
python build_db.py
```

after placing PDF files inside the `data/` folder.

---

## Author

Khadija Nazir

BS Computer Science
