import os
import streamlit as st

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI RAG Chatbot")
st.caption("Retrieval-Augmented Generation using ChromaDB + Groq")

# ====================================
# SIDEBAR
# ====================================

st.sidebar.title("⚙️ Settings")

st.sidebar.info("""
This chatbot answers questions from your indexed PDF documents.

Features:
✅ ChromaDB
✅ LangChain
✅ Groq Llama 3
✅ RAG Pipeline
""")

api_key = st.sidebar.text_input(
    "🔑 Groq API Key",
    type="password"
)

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# ====================================
# DATABASE
# ====================================

CHROMA_DIR = "/content/drive/MyDrive/RAG_10K_Project/chroma_db_v2"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(
    search_kwargs={"k":3}
)

# ====================================
# PROMPT
# ====================================

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer is not available in the context,
reply exactly:

I don't know based on the provided documents.

Context:
{context}

Question:
{question}
""")

# ====================================
# CHAT HISTORY
# ====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ====================================
# USER QUESTION
# ====================================

question = st.chat_input("💬 Ask anything from your documents...")

if question:

    if api_key == "":
        st.error("⚠️ Please enter your Groq API Key.")
        st.stop()

    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    with st.spinner("🔍 Searching documents and generating answer..."):

        docs = retriever.invoke(question)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        chain = prompt | llm

        response = chain.invoke({
            "context":context,
            "question":question
        })

    answer = response.content

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":answer
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.divider()

    st.subheader("📚 Source Documents")

    if len(docs)==0:

        st.warning("No relevant documents found.")

    else:

        for i, doc in enumerate(docs, start=1):

            with st.expander(f"📄 Source {i}"):

                if "source" in doc.metadata:
                    st.write(
                        "**File:**",
                        os.path.basename(doc.metadata["source"])
                    )

                if "page" in doc.metadata:
                    st.write(
                        "**Page:**",
                        doc.metadata["page"]+1
                    )

                st.write("---")

                st.write(doc.page_content[:700])

# ====================================
# FOOTER
# ====================================

st.divider()

st.caption("Built with ❤️ using Streamlit • LangChain • ChromaDB • Groq")