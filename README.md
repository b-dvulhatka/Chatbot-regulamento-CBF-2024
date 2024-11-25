# Chatbot com Regulamento CBF 2024

This project is a chatbot application designed to interact with the **Regulamento CBF 2024** or other PDFs, answering user queries using OpenAI's GPT models and LangChain. The application leverages vector-based storage for efficient document retrieval.

## Project Overview

The project consists of two main components:

1. **`app.py`:** The Streamlit-powered frontend for the chatbot interface.
2. **`vectorstore.py`:** A utility script responsible for processing and vectorizing PDF content for retrieval.

### Features

- **Chat Interface:** Users can ask questions related to the contents of a loaded PDF file.
- **Vector-Based Retrieval:** Uses FAISS to efficiently retrieve relevant sections of the document.
- **Session Management:** Chat history is maintained within the session for a seamless experience.
- **Secure API Integration:** OpenAI API key is securely accessed from `.streamlit/secrets.toml`.

---

## Files

### `app.py`

This is the main file for the Streamlit application. It serves as the user interface for the chatbot.

#### Key Functionalities app.py

1. **Chatbot Interface:**
   - Users input questions related to the document.
   - Responses are generated using the `gpt-3.5-turbo` model from OpenAI.
2. **PDF Loading:**
   - Automatically loads the first PDF file from the `data/` directory.
3. **Session State:**
   - Maintains chat history using `st.session_state` to display past interactions during a session.
4. **Error Handling:**
   - Warns users if no PDF is found in the directory.
   - Displays an error if the OpenAI API key is not set.

#### Key Sections of Code

- **PDF Loading:**
  Automatically selects the first available PDF in the `data/` folder:

  ```python
  pdf_files = [f for f in os.listdir("data/") if f.endswith(".pdf")]
  if pdf_files:
      pdf_path = os.path.join("data", pdf_files[0])
  ```

- **Chatbot Logic:**
  Uses LangChain's `RetrievalQA` to connect GPT with the FAISS vectorstore:

  ```python
  qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
  response = qa_chain.run(user_question)
  ```

- **Session State for History:**
  Appends user questions and bot responses to the session state:

  ```python
  st.session_state["messages"].append({"role": "user", "content": user_question})
  st.session_state["messages"].append({"role": "assistant", "content": response})
  ```

---

### `vectorstore.py`

This utility file is responsible for processing PDF files and creating a vector-based storage system for document retrieval.

#### Key Functionalities vectorstore.py

1. **PDF Content Extraction:**
   - Reads the contents of a PDF file.
2. **Embedding Generation:**
   - Converts PDF content into vector embeddings using OpenAI's embedding model.
3. **FAISS Vectorstore Creation:**
   - Stores and retrieves document embeddings efficiently using FAISS.

#### Key Sections of Code vectorstore.py

- **PDF Reading:**
  Extracts text from a given PDF file:

  ```python
  from PyPDF2 import PdfReader
  reader = PdfReader(pdf_path)
  text = " ".join([page.extract_text() for page in reader.pages])
  ```

- **Embedding Conversion:**
  Generates vector embeddings for the extracted text using OpenAI embeddings:

  ```python
  from langchain.embeddings.openai import OpenAIEmbeddings
  embeddings = OpenAIEmbeddings(openai_api_key=api_key)
  ```

- **FAISS Indexing:**
  Creates and saves the FAISS index:

  ```python
  from langchain.vectorstores import FAISS
  vectorstore = FAISS.from_texts([text], embedding=embeddings)
  ```

---

## How It Works

1. **Load PDF:** The app automatically loads the first PDF file in the `data/` folder.
2. **Create Vectorstore:** The content of the PDF is converted into vector embeddings and stored in FAISS.
3. **User Input:** Users can ask questions about the PDF through the chat interface.
4. **Response Generation:** The app retrieves the most relevant sections of the document using the vectorstore and generates a response using OpenAI's GPT model.
5. **History Maintenance:** The app maintains a record of the user's questions and the chatbot's responses for the duration of the session.

---
