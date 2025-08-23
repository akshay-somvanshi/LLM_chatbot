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
import sqlite3

# Database initialisation for storing conversations
############################################

# Interact with SQLite database held in file conversations.db
connection = sqlite3.connect("conversations.db")

# Object that allows to send SQL statements 
cursor = connection.cursor()

# Database design: 
# - Conversation table [conversationId(Primary id), user_name(User who created the conversation) and title(First message of conversation)]
# - Messages table [conversationId(Foreign id), message_id(Primary id), sender(AI or user), content(the message) and Timestamp]

def init_db():
    """
    Create the tables if they dont exist
    """
    cursor.execute("""CREATE TABLE IF NOT EXISTS conversations (
                   ConversationId INTEGER PRIMARY KEY AUTOINCREMENT,
                   user_name TEXT NOT NULL,
                   title TEXT 
                    )
                """)
    
    cursor.execute("""CREATE TABLE IF NOT EXISTS messages(
                   message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                   ConversationId INTEGER,
                   sender TEXT NOT NULL,
                   content TEXT NOT NULL,
                   time DATETIME DEFAULT CURRENT_TIMESTAMP,
                   FOREIGN KEY(ConversationId) REFERENCES conversations(ConversationId) 
                    )
                """)
    
    connection.commit()

def create_conversation(user_name, title):
    """
    Create new conversation with the given user
    """
    # ? is the placeholder for variables
    cursor.execute("INSERT INTO conversations (user_name, title) VALUES (?, ?)", 
                   user_name, title)
    
    connection.commit()
    return cursor.lastrowid  # Returns the auto-generated ID

def save_message(conversation_id, content, sender):
    cursor.execute("INSERT INTO messages (conversation_id, sender, content) VALUES (?, ?, ?)",
                   conversation_id, sender, content)
    
    connection.commit()

def load_conversation_messages(conversation_id):
    """
    Load all messages for a conversation
    """
    cursor.execute("""
                   SELECT content, sender, timestamp 
                   FROM messages where ConversationId = ?
                   ORDER BY timestamp """, conversation_id)
    
    return cursor.fetchall()

def load_user_conversation(user_name):
    """
    Load all conversations for the user
    """
    cursor.execute("SELECT ConversationId, title from conversations WHERE user_name = ?", user_name)

    return cursor.fetchall()

init_db()

def debug_show_all_conversations():
    """Debug function to see all conversations"""
    cursor.execute("SELECT * FROM conversations")
    results = cursor.fetchall()
    st.write("All conversations:", results)

def debug_show_all_messages():
    """Debug function to see all messages"""
    cursor.execute("SELECT * FROM messages")
    results = cursor.fetchall()
    st.write("All messages:", results)

def debug_test_database():
    """Test creating conversation and message"""
    # Test creating a conversation
    conv_id = create_conversation("Alice Johnson", "Test Conversation")
    st.write(f"Created conversation with ID: {conv_id}")
    
    # Test saving a message
    save_message(conv_id, "user", "Hello, this is a test message")
    save_message(conv_id, "assistant", "Hi! This is a test response")
    
    # Test loading messages
    messages = load_conversation_messages(conv_id)
    st.write("Messages:", messages)

############################################

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

# Only create if directory doesn't exist
if not os.path.exists("vectorstore_db"):
    vectorstore_db = Chroma.from_documents(
        persist_directory="vectorstore_db",
        documents=retrieve_data(["data/User_data.json"]),
        embedding=embedding_model,
    )
else:
    # Just load the existing database
    vectorstore_db = Chroma(
        persist_directory="vectorstore_db", 
        embedding_function=embedding_model
    )

# Interface that returns documents given an unstructured query - select the top 3 documents
vector_retriever = vectorstore_db.as_retriever(search_kwargs={"k": 3})

############################################

cloud_box = st.checkbox(
    "Use Cloud LLM",
    value = False
)

debug_db = st.checkbox(
    "Debug db",
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

        # Searching the database with the user prompt3
        relevant_info = vectorstore_db.similarity_search(search_query, k=5)

        # Filter to current user only
        retrieved_data = []
        for doc in relevant_info:
            if current_user in doc.page_content:
                # Extract the text content from the search result
                retrieved_data.append(doc.page_content)
                if len(retrieved_data) >= 3:  # Stop after finding 3 user-specific docs
                    break

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

        if debug_db:
            debug_test_database()
            debug_show_all_conversations()
            debug_show_all_messages()
