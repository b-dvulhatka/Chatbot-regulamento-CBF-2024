from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

def get_pdf_retriever(pdf_paths, api_key):
    """
    Load and embed multiple PDFs into a single retriever.

    Args:
        pdf_paths (list): A list of paths to the PDF files.
        api_key (str): The OpenAI API key.

    Returns:
        FAISS retriever object.
    """
    all_docs = []
    # Iterate over each PDF path in the list
    for path in pdf_paths:
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            # Add the source file name to the metadata of each document
            for doc in docs:
                doc.metadata['source'] = os.path.basename(path)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading file {path}: {e}")
            # Skip to the next file if one is corrupted
            continue

    if not all_docs:
        raise ValueError("No documents could be loaded. Please check the PDF files.")

    # Split all combined documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    # Create embeddings using the provided API key
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Create the vector store from the chunks and embeddings
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Return the vector store as a retriever
    return vectorstore.as_retriever()
