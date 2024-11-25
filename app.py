import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from utils.vectorstore import load_pdf_and_create_vectorstore
import os

# Streamlit app layout
st.title("Chatbot com Regulamento CBF 2024")

# Retrieve the OpenAI API key from secrets
api_key = st.secrets["OpenAI_key"]

if api_key:
    # Initialize session state for history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # List to store message history

    # Get the first PDF in the "data" folder
    pdf_files = [f for f in os.listdir("data/") if f.endswith(".pdf")]
    if pdf_files:
        pdf_path = os.path.join("data", pdf_files[0])  # Automatically load the first PDF

        # Load vectorstore
        vectorstore = load_pdf_and_create_vectorstore(pdf_path, api_key)

        # Initialize GPT model and Retrieval QA
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

        # Chat interface
        user_question = st.text_input("Pergunte sobre o regulamento:")

        if user_question:
            # Append user question to session state
            st.session_state["messages"].append({"role": "user", "content": user_question})

            # Get the response from the QA chain
            response = qa_chain.run(user_question)

            # Append response to session state
            st.session_state["messages"].append({"role": "assistant", "content": response})

        # Display chat history only if it exists
        if st.session_state["messages"]:
            st.subheader("Histórico")
            for msg in st.session_state["messages"]:
                if msg["role"] == "user":
                    st.write(f"**Você:** {msg['content']}")
                elif msg["role"] == "assistant":
                    st.write(f"**Bot:** {msg['content']}")
    else:
        st.warning("Nenhum PDF encontrado na pasta `data/`. Por favor, adicione um arquivo PDF.")
else:
    st.error("API Key not found! Please add it to .streamlit/secrets.toml.")
