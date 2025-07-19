from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF
loader = PyMuPDFLoader("Suman_Kumari_Resume.pdf")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load PDF
loader = PyMuPDFLoader("Suman_Kumari_Resume.pdf")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

from langchain.chat_models import ChatGroq
from langchain.chains import RetrievalQA

# Use Groq + Llama 3
llm = ChatGroq(
    temperature=0.2,
    model_name="llama3-8b-8192",  # or use llama3-70b if needed
    api_key="YOUR_GROQ_API_KEY"  # sign up at groq.com
)

# Build retriever
retriever = db.as_retriever()

# Connect retriever + LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


# save as app.py
import streamlit as st

st.title("Gen AI Q&A")
query = st.text_input("Ask something:")

if query:
    response = qa_chain.run(query)
    st.write(response)