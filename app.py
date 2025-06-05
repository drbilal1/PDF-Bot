import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from tempfile import NamedTemporaryFile
import google.generativeai as genai

# Set Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyC3iQpf0g8KHcXaNQ4RCioG6e_j8lzryyo"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0,
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# Streamlit UI
st.title("📄 Document Q&A Bot with Gemini")
st.write("Upload PDFs and ask questions about their content")

# File upload
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Initialize components
@st.cache_resource
def process_documents(files):
    documents = []
    for file in files:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            documents.extend(loader.load())
        os.unlink(tmp.name)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(texts, embeddings)

    return db

# Question answering
if uploaded_files:
    db = process_documents(uploaded_files)
    st.success(f"Processed {len(uploaded_files)} PDF(s). You can now ask questions!")

    # Initialize Gemini with correct parameters
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

    # Check your prompt template for invalid fields (e.g., 'thought')
    prompt = """Answer the question based on the context:
    Context: {context}
    Question: {query}
    Answer:"""  # No 'thought' field here!

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

    # Question input
    question = st.text_input("Ask a question about the document(s):")
    if question:
        with st.spinner("Thinking..."):
            result = qa_chain({"query": question})
            st.subheader("Answer")
            st.write(result["result"])
else:
    st.info("Please upload PDF files to begin")