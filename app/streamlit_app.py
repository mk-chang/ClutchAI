# Description: A simple Streamlit app that uses ClutchAI Agent for Yahoo Fantasy Basketball.
# Local Testing Command: streamlit run streamlit_app.py
# For Deployment: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/llm-quickstart

import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Add project root to Python path so we can import ClutchAI
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.resolve()

# Verify project root by checking for ClutchAI directory
if not (project_root / "ClutchAI").exists():
    # Try current working directory as fallback
    cwd_root = Path.cwd().resolve()
    if (cwd_root / "ClutchAI").exists():
        project_root = cwd_root
    else:
        raise FileNotFoundError(
            f"ClutchAI directory not found. Expected at: {project_root / 'ClutchAI'}"
        )

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from ClutchAI.agent import ClutchAIAgent

# Load environment variables from .env file
env_file_location = project_root
env_file_path = env_file_location / ".env"

# Load .env file if it exists
if env_file_path.exists():
    load_dotenv(dotenv_path=env_file_path, override=True)
else:
    # Try loading from default location (current directory)
    load_dotenv(override=True)

# Set Streamlit page configuration
st.title("🏀 ClutchAI 🤖")
st.caption("""
    🚀 Your AI-powered fantasy sports assistant. \n\n 
    💬 Get real-time, context-aware insights like start/sit advice, performance summaries, and trade recommendations.
""")

# Initialize session state with environment variables if not already set
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.environ.get('OPENAI_API_KEY', "")
if "yahoo_client_id" not in st.session_state:
    st.session_state.yahoo_client_id = os.environ.get('YAHOO_CLIENT_ID', "")
if "yahoo_secret" not in st.session_state:
    st.session_state.yahoo_secret = os.environ.get('YAHOO_CLIENT_SECRET', "")
if "yahoo_league_id" not in st.session_state:
    st.session_state.yahoo_league_id = os.environ.get('YAHOO_LEAGUE_ID', "58930")

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", 
        key="openai_api_key", 
        type="password", 
        help="Set OPENAI_API_KEY"
    )
    yahoo_client_id = st.text_input(
        "Yahoo Client ID", 
        key="yahoo_client_id", 
        type="password", 
        help="Set YAHOO_CLIENT_ID"
    )
    yahoo_secret = st.text_input(
        "Yahoo Secret", 
        key="yahoo_secret", 
        type="password", 
        help="Set YAHOO_CLIENT_SECRET"
    )
    yahoo_league_id = st.text_input(
        "Yahoo League ID", 
        key="yahoo_league_id", 
        placeholder="58930",
        help="Set YAHOO_LEAGUE_ID"
    )
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[Get an Yahoo API keys](https://developer.yahoo.com/apps/)"
    "[View the source code](https://github.com/mk-chang/ClutchAI)"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Initialize agent in session state if not already initialized
if "agent" not in st.session_state:
    st.session_state["agent"] = None
if "agent_key" not in st.session_state:
    st.session_state["agent_key"] = None

# Initialize or reinitialize agent on page load if credentials are available
if openai_api_key and yahoo_league_id:
    try:
        league_id_int = int(yahoo_league_id)
    except ValueError:
        st.error("Yahoo League ID must be a number.")
        st.stop()
    
    # Check if agent needs to be initialized or reinitialized
    agent_key = f"{openai_api_key[:10]}_{yahoo_client_id[:10] if yahoo_client_id else 'none'}_{league_id_int}"
    if st.session_state.get("agent_key") != agent_key or st.session_state["agent"] is None:
        with st.spinner("Initializing ClutchAI Agent..."):
            try:
                st.session_state["agent"] = ClutchAIAgent(
                    yahoo_league_id=league_id_int,
                    yahoo_client_id=yahoo_client_id or None,
                    yahoo_client_secret=yahoo_secret or None,
                    openai_api_key=openai_api_key,
                    env_file_location=env_file_location,
                )
                st.session_state["agent_key"] = agent_key
            except Exception as e:
                st.error(f"Failed to initialize agent: {e}")
                st.session_state["agent"] = None
                st.session_state["agent_key"] = None

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    
    if not yahoo_league_id:
        st.info("Please add your Yahoo League ID to continue.")
        st.stop()
    
    if st.session_state["agent"] is None:
        st.error("Agent not initialized. Please check your credentials.")
        st.stop()
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Get agent response
    with st.spinner("Thinking..."):
        try:
            # Pass conversation history (excluding the current prompt which is already added)
            conversation_history = st.session_state.messages[:-1]  # All messages except the one we just added
            response = st.session_state["agent"].chat(prompt, conversation_history=conversation_history)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.chat_message("assistant").write(error_msg)