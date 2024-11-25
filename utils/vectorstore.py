from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

def load_pdf_and_create_vectorstore(pdf_path: str, openai_api_key: str) -> FAISS:
    # Load the PDF file
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create a FAISS vectorstore
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore
