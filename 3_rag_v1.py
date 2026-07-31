import os
from dotenv import load_dotenv

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

os.environ['LANGCHAIN_PROJECT'] = 'RAG Chatbot'

load_dotenv()

PDF_PATH = "islr.pdf"

# 1. Load PDF
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

splits = splitter.split_documents(docs)

# 3. Embeddings (HuggingFace)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Vector Store
vectorstore = FAISS.from_documents(splits, embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 5. Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer ONLY from the provided context. "
        "If the answer is not in the context, say 'I don't know.'"
    ),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}"
    )
])

# 6. Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# 7. Format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parallel = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

# 8. RAG Chain
chain = (
    parallel | prompt | llm | StrOutputParser()
)

# 9. Ask questions
print("PDF RAG Ready! (Press Ctrl+C to exit)")

while True:
    question = input("\nQ: ")

    if question.lower() in ["exit", "quit"]:
        break

    answer = chain.invoke(question)

    print("\nA:", answer)