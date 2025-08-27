import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.globals import set_verbose
set_verbose(False)  # disable verbose warnings
from utils.vectorstore import get_pdf_retriever

st.set_page_config(page_title="Chatbot IA Regulamento CBF")
st.title("Chatbot com Regulamento CBF 2024")

# -----------------------------
# API Key
# -----------------------------
api_key = st.secrets.get("OpenAI_key")
if not api_key:
    st.error("Chave da API da OpenAI não encontrada! Adicione-a em .streamlit/secrets.toml.")
    st.stop()

# -----------------------------
# Conversation History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# PDF Retriever
# -----------------------------
# 1. List all PDF files in the 'data/' directory
pdf_files = [f for f in os.listdir("data/") if f.endswith(".pdf")]
if not pdf_files:
    st.warning("No PDF files found in the `data/` folder. Please add at least one PDF.")
    st.stop()

# 2. Create a list of full paths for each PDF file
pdf_paths = [os.path.join("data", f) for f in pdf_files]

# Display the loaded files
st.info(f"Loaded PDF files: {', '.join(pdf_files)}")

# The cache function needs a hashable (immutable) argument.
# Lists are mutable, so we convert our list of paths to a tuple.
@st.cache_resource
def load_retriever(paths_tuple, key):
    """
    Loads the retriever. The @st.cache_resource decorator ensures this function
    only runs once for the same arguments, preventing reprocessing of the PDFs
    on each user interaction.
    """
    # We convert the tuple back to a list to pass it to the actual function
    return get_pdf_retriever(list(paths_tuple), key)

# 3. Call the cache function with the tuple of paths and the API key
retriever = load_retriever(tuple(pdf_paths), api_key)

# -----------------------------
# Language Model (streaming)
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, streaming=True)

# -----------------------------
# Prompt template
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
Você é um assistente especialista na Diretriz Técnica do Futebol da CBF. Sua tarefa é analisar o contexto fornecido e responder à pergunta do usuário com máxima precisão.

Siga estas regras rigorosamente:
1.  Sua resposta deve ser baseada única e exclusivamente no texto presente no "Contexto".
2.  Responda de forma direta e clara à pergunta do usuário.
3.  Após a resposta direta, se a informação foi encontrada, adicione uma seção "Fonte" e cite o trecho exato do contexto que comprova sua resposta.
4.  Se a informação não estiver de forma alguma no contexto, não tente inferir ou adivinhar. Apenas responda: "A informação solicitada não foi encontrada na diretriz técnica fornecida."

Contexto:
{context}

Pergunta do Usuário: {input}
""")

# -----------------------------
# Runnable pipeline (retriever → prompt → LLM)
# -----------------------------
retrieval_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
)

# -----------------------------
# Chat Interface
# -----------------------------
user_question = st.chat_input("Pergunte sobre o regulamento:")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Stream tokens to the chat
    with st.chat_message("assistant"):
        response_stream = retrieval_chain.stream(user_question)
        answer = st.write_stream(response_stream)

    # Store the final response in history
    st.session_state.messages.append({"role": "assistant", "content": answer})

# -----------------------------
# Display History
# -----------------------------
st.subheader("Histórico")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
