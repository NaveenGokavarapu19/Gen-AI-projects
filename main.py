# main.py - Backend Logic

# ── IMPORTS ────────────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
import tempfile

# ── EMBEDDINGS + LLM SETUP ─────────────────────────────────────────────
def get_embeddings():
    return BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name="us-east-1"
    )

def get_llm():
    return ChatBedrockConverse(
        model="amazon.nova-2-lite-v1:0",
        region_name="us-east-1",
        temperature=0,
        max_tokens=512
    )

# ── VECTOR STORE ────────────────────────────────────────────────────────
def build_vector_store(uploaded_file, embeddings):
    """
    Takes an uploaded PDF file object and returns a FAISS vector store.
    Steps: Save temp → Load → Chunk → Embed → Return
    """
    # Save uploaded file to temp location so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)

    # Embed + store in FAISS
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Clean up temp file
    os.unlink(tmp_path)

    return vector_store, len(pages), len(chunks)

def load_vector_store(embeddings):
    """
    Loads FAISS index from disk if it exists.
    Used for local testing via terminal.
    """
    if os.path.exists("faiss_index"):
        return FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None

# ── RAG CHAIN ───────────────────────────────────────────────────────────
def build_rag_chain(vector_store, llm):
    """
    Builds and returns the RAG chain.
    Takes vector_store and llm as inputs.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant.
         Answer the question based only on the context provided.
         If the answer is not in the context, say 'I don't know'.

         Context: {context}"""),
        ("human", "{input}")
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(
        vector_store.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain
    )
    return rag_chain

# ── ASK QUESTION ────────────────────────────────────────────────────────
def ask_question(rag_chain, question):
    """
    Takes a rag_chain and question string.
    Returns answer and sources.
    """
    response = rag_chain.invoke({"input": question})
    answer = response["answer"]
    sources = [
        {
            "page": doc.metadata.get("page"),
            "text": doc.page_content[:150]
        }
        for doc in response["context"]
    ]
    return answer, sources