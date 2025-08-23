from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import json
import os
import time

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
    model = "deepseek/deepseek-r1-0528-qwen3-8b:free",
    api_key = API_Key,
    base_url = Remote_base_url
)

# Data Augmentation
############################################

# Embed all data 
def retrieve_data(docs):
    all_documents = []  # Accumulate all documents
    for file in docs:   
        with open(file, "r") as f:
            data = json.loads(f.read())
            splitter = RecursiveJsonSplitter(max_chunk_size=900)
            documents = splitter.create_documents(texts=data, convert_lists=True)
            all_documents.extend(documents)  # Add to collection
    return all_documents

embedding_model = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2'
)

# Store it in a vector database
vectorstore_db = Chroma.from_documents(
    persist_directory="vectorstore_db",
    documents=retrieve_data(["data/User_data.json"]),
    embedding=embedding_model,
)

vectordb = Chroma(
    persist_directory="vectorstore_db", embedding_function=embedding_model
)

# Interface that returns documents given an unstructured query - select the top 3 documents
vector_retriever = vectordb.as_retriever(search_kwargs={"k": 3})

############################################

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

st.session_state.setdefault(
    "user",
    # No user empty string initialisation
    ""
)

# Set the UI title based on whether user name is known 
if st.session_state["user"] == "":
    st.title("Enter your full name")
    prompt = st.chat_input("Enter your name here")
    if prompt:
        # Store the user name in the session state so we can use it for data retrieval
        st.session_state["user"] = prompt
        # Refresh to show new interface
        st.rerun()
else:
    st.title(f"Hello {st.session_state["user"]}! How can I help?")

    # Iterate through the messages and print them on the console
    for msg in st.session_state["messages"]:
        # Print based on role and content
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Type message here")

    def stream_data(response):
        for word in response.split(" "):
            yield word + " "
            time.sleep(0.02)

    if prompt:
        # Filter to only include current user's data
        current_user = st.session_state["user"]

        # Append the query with the user information to retrieve the correct data
        search_query = f"User: {st.session_state['user']}. Question: {prompt}"

        # Searching the database with the user prompt
        relevant_info = vector_retriever.get_relevant_documents(search_query)

        # Extract the text content from the search result
        retrieved_data = [data.page_content for data in relevant_info]

        # System message for priming AI behavior.
        message = [{"role": "system", "content": f"You are a helpful assistant. When appropriate, provide planning-oriented responses with actionable steps \
                    with key insights."}]
        
        if retrieved_data:
            st.sidebar.write("Retrieved Context:")
            for i, data in enumerate(retrieved_data, 1):
                st.sidebar.write(f"{i}. {data}")
            content_data = "\n".join(retrieved_data)
            message.append({
                "role": "system",
                "content" : f"\n Relevant user information:\n{content_data}"
            })

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
            message.append({
                "role": msg["role"],
                "content": msg["content"]
            })

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
            st.write_stream(stream_data(response.content))
            # st.write(response.content)
