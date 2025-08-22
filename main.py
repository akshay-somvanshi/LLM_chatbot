from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import json
import os

# Load From .env File
API_Key = os.environ.get("API_Key")
Local_base_url = os.environ.get("Local_base_url")
Remote_base_url = os.environ.get("Remote_base_url")

local_llm = ChatOpenAI(
    # Same model as in Dockerfile
    model = "ai/qwen2.5",
    # Even though it is not required we must pass the argument
    api_key = "No",
    base_url = Local_base_url
)

cloud_llm = ChatOpenAI(
    model = "openai/gpt-oss-20b:free",
    api_key = API_Key,
    base_url = Remote_base_url
)

# Data Augmentation
############################################

def retrieve_data(docs):
    all_documents = []  # Accumulate all documents
    for file in docs:   
        with open(file, "r") as f:
            data = json.loads(f.read())
            splitter = RecursiveJsonSplitter(max_chunk_size=300)
            documents = splitter.create_documents(texts=data, convert_lists=True)
            all_documents.extend(documents)  # Add to collection
    return all_documents

embedding_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2'
)

vectorstore_db = Chroma.from_documents(
    persist_directory="vectorstore_db",
    documents=retrieve_data(["data/User_data.json"]),
    embedding=embedding_model,
)

vectordb = Chroma(
    persist_directory="vectorstore_db", embedding_function=embedding_model
)
vector_retriever = vectordb.as_retriever()


############################################

# Set the UI title
st.title("LLM ChatBot")

cloud_box = st.checkbox(
    "Use Cloud LLM",
    value = False
)

# User session to store conversations
st.session_state.setdefault(
    "messages",
    # Empty message list initialisation
    []
)

# System message for priming AI beh avior.
message = [{"role": "system", "content": "You are a helpful assistant.When appropriate, provide planning-oriented responses with actionable steps \
             with key insights."}]

# Iterate through the messages and print them on the console
for msg in st.session_state["messages"]:
    # Print based on role and content
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Type message here")

if prompt:

    # Store the user messages in a dictionary
    st.session_state["messages"].append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Pre-built user role
    with st.chat_message("user"):
        st.write(prompt)

    # For the chatbot to remember conversations
    for msg in st.session_state["messages"]:
        message.append({"role": msg["role"], "content": msg["content"]})

    if cloud_box:
        llm = cloud_llm
    else:
        llm = local_llm

    response = llm.invoke(message)

    # Store the LLM messages in a dictionary
    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": response.content
        }
    )

    # Pre-built assistant role
    with st.chat_message("assistant"):
        st.write(response.content)
