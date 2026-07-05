import os
import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI RAG Chatbot")
st.caption("Retrieval-Augmented Generation using ChromaDB + Groq")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("⚙️ Settings")

st.sidebar.success("✅ ChromaDB")
st.sidebar.success("✅ LangChain")
st.sidebar.success("✅ Groq Llama 3")
st.sidebar.success("✅ HuggingFace Embeddings")

api_key = os.getenv("GROQ_API_KEY")

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# =====================================
# LOAD CHROMA DATABASE
# =====================================

CHROMA_DIR = "chroma_db_v2"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model
)

db_count = vector_db._collection.count()

st.sidebar.info(f"Indexed Chunks: {db_count}")

retriever = vector_db.as_retriever(
    search_kwargs={"k": 5}
)

# =====================================
# PROMPT
# =====================================

prompt = ChatPromptTemplate.from_template(
"""
You are an AI assistant.
Answer ONLY using the provided context.
If the answer cannot be found inside the context, reply exactly:
I don't know based on the provided documents.
Context:
{context}
Question:
{question}
"""
)

# =====================================
# CHAT HISTORY
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # =====================================
# USER INPUT
# =====================================

question = st.chat_input("💬 Ask a question from your documents...")

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    if not api_key:
        st.error("❌ GROQ_API_KEY not found.")
        st.stop()

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    with st.spinner("🔍 Searching documents..."):

        docs = retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        chain = prompt | llm

        response = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        answer = response.content

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)
        # =====================================
# SOURCE DOCUMENTS
# =====================================

    st.divider()
    st.subheader("📚 Source Documents")

    if len(docs) == 0:

        st.warning("No relevant documents found.")

    else:

        for i, doc in enumerate(docs, start=1):

            with st.expander(f"📄 Source {i}"):

                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", None)

                st.write("**File:**", os.path.basename(source))

                if page is not None:
                    st.write("**Page:**", page + 1)

                st.markdown("---")

                st.write(doc.page_content[:800])

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "❤️ Built with Streamlit • LangChain • ChromaDB • Hugging Face • Groq"
)
