import os
from dotenv import load_dotenv

from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PDF_PATH = "islr.pdf"


# ---------------- Load PDF ----------------

@traceable(name="load_pdf", tags=['pdf','loader'],metadata={'loader' : 'PyPDFLoader'})
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()


# ---------------- Split ----------------

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


# ---------------- Build Vector Store ----------------

@traceable(name="build_vectorstore", tags=['vectorstore','faiss'], metadata={'vectorstore' : 'faiss', 'embedding' : 'huggingface'})
def build_vectorstore(splits):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        splits,
        embeddings
    )

    return vectorstore


# ---------------- Setup ----------------

@traceable(name="setup_pipeline")
def setup_pipeline(pdf_path: str):

    docs = load_pdf(pdf_path)

    splits = split_documents(docs)

    vectorstore = build_vectorstore(splits)

    return vectorstore


# ---------------- Groq LLM ----------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


# ---------------- Prompt ----------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer ONLY from the provided context. "
            "If the answer is not in the context, say 'I don't know.'",
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}",
        ),
    ]
)


# ---------------- Format Docs ----------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------- Build Pipeline ----------------

vectorstore = setup_pipeline(PDF_PATH)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

parallel = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

chain = (
    parallel
    | prompt
    | llm
    | StrOutputParser()
)


# ---------------- Ask Question ----------------

print("PDF RAG Ready! (Ctrl+C to Exit)")

while True:

    question = input("\nQ: ")

    if question.lower() in ["exit", "quit"]:
        break

    config = {
        "run_name": "pdf_rag_query"
    }

    answer = chain.invoke(
        question,
        config=config,
    )

    print("\nA:", answer)