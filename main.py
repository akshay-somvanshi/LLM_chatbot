from langchain_openai import ChatOpenAI
import streamlit as st
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

######################################

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

# message = [{"role": "system", "content": "Respond with actionable, planning-oriented steps rather than only providing descriptive information."}]

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

    context = ""

    # For the chatbot to remember conversations
    for msg in st.session_state["messages"]:
        context += msg["role"] + ": " + msg["content"]

    if cloud_box:
        llm = cloud_llm
    else:
        llm = local_llm

    response = llm.invoke(context)

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
