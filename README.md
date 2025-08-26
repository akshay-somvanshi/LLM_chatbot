# Planning-Oriented LLM ChatBot

A conversational AI application that provides personalized, actionable planning advice through retrieval-augmented generation (RAG) and conversation management.

## Features

### Core Functionality
- **Planning-Oriented Responses**: System prompt engineered to provide actionable steps and concrete recommendations
- **Retrieval-Augmented Generation (RAG)**: Integrates user-specific data for personalized responses
- **Multi-User Support**: User-specific conversations and data retrieval
- **Conversation Persistence**: SQLite database storage for chat history
- **Real-time Streaming**: Word-by-word response streaming for better UX

### Advanced Features
- **Dual LLM Support**: Switch between local and cloud models ("Think Deeper" option)
- **Semantic Search**: Vector embeddings with ChromaDB for intelligent data retrieval
- **Conversation Management**: Sidebar interface for browsing and switching between chats

## Architecture

```
Frontend (Streamlit) ←→ Backend Logic ←→ Vector Database (ChromaDB)
                              ↓
                        SQLite Database
                      • Conversations
                      • Messages
                              ↓
                        LLM APIs
                    • Local (Qwen2.5)
                    • Cloud (DeepSeek)
```

## Prerequisites

- Python 3.12+
- Docker 4.44.3
- LLM API access

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd llm-chatbot
```

### 2. Environment Configuration
Create a .env file with your API credentials:
```bash
API_Key=your_api_key_here
Local_base_url=your_local_api_url (Docker model runner for example)
Remote_base_url=your_remote_api_url
```
### 2. Docker Setup
Make sure Docker version is at 4.44.3 (otherwise we have "additional property provider not allowed" error)

Build and run the application:
```bash
docker compose up --build
```

### 3. Access the Application
Navigate to: `http://localhost:8501`

## Sample Data

The application includes pre-tested user profiles for demonstration:
- Alice Johnson
- Bob Smith

**Try logging in as either user to explore pre-existing conversations and see personalized responses based on their profiles!**

## Usage

### Getting Started
1. **Enter Your Name**: Use one of the users from User_data.json or add a new user in the file.
2. **Start Chatting**: Ask planning-oriented questions like:
   - "Help me plan my next project"
   - "What programming languages should I learn?"

### Key Features to Try
- **Switch Conversations**: Use the sidebar to browse previous chats
- **New Conversations**: Start fresh discussions anytime
- **Retrieved Context**: Toggle to see what user data influenced responses
- **Model Selection**: Try both local and cloud models

## Technical Implementation

### Database Schema
```sql
-- Conversations table
CREATE TABLE conversations (
    ConversationId INTEGER PRIMARY KEY AUTOINCREMENT,
    user_name TEXT NOT NULL,
    title TEXT
);

-- Messages table
CREATE TABLE messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ConversationId INTEGER,
    sender TEXT NOT NULL,
    content TEXT NOT NULL,
    time DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(ConversationId) REFERENCES conversations(ConversationId)
);
```

### RAG Pipeline
1. **Document Processing**: JSON user data chunked and embedded
2. **Vector Storage**: ChromaDB with sentence-transformers embeddings
3. **Semantic Retrieval**: User-aware similarity search
4. **Context Injection**: Relevant data added to system prompt
5. **LLM Generation**: Planning-oriented response with user context

### System Prompt Design
The AI is specifically prompted to provide:
- **Key Insights**: Brief analysis of the situation
- **Action Steps**: Specific, numbered steps to take
- **Next Steps**: Follow-up recommendations

## Project Structure

```
.
├── main.py                 # Main application file
├── data/
│   └── User_data.json     # Sample user profiles and data
├── docker-compose.yml     # Container orchestration
├── Dockerfile            # Application container
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Advanced Configuration

### Adding New Users
Edit `data/User_data.json` to include new user profiles:
```json
{
    "user": {"name": "Your Name", "role": "Your Role"},
    "skills": {"technical": [...], "soft": [...]},
    "projects": [...],
    "goals": {...}
}
```

### Model Configuration
- **Local Model**: Configured for Qwen2.5 via Docker service
- **Cloud Model**: DeepSeek API (configurable in environment variables)
- **Embeddings**: HuggingFace sentence-transformers (all-MiniLM-L6-v2)

## Troubleshooting

### Common Issues
- **Docker compose error**: Update docker to version 4.44.3
- **Database Locked**: Stop Docker containers before deleting vector database
- **API Errors**: Verify API key is valid and LLM model is available

### Reset Instructions
```bash
# Clean reset
docker compose down
sudo rm -rf vectorstore_db conversations.db
docker compose up --build
```

## License

This project is open source and available under the MIT License.

## Resources

- **LangChain**: For LLM orchestration and RAG pipeline
- **Streamlit**: For rapid web application development
- **ChromaDB**: For vector storage and similarity search
- **HuggingFace**: For embedding models and transformers

---
