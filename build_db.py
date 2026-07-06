import os
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_PATH = "data"
DB_PATH = "chroma_db_v2"

# Delete old database
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)

print("Loading PDFs...")

loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()

print(f"Loaded {len(documents)} pages.")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating Chroma Database...")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=DB_PATH
)

print("Database Created Successfully!")
print("Total Chunks:", db._collection.count())