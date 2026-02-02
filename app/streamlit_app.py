# Description: A simple Streamlit app that uses ClutchAI Agent for Yahoo Fantasy Basketball.
# Local Testing Command: streamlit run app/streamlit_app.py
# Debug Mode: streamlit run app/streamlit_app.py -- --debug
#   OR: CLUTCHAI_DEBUG=1 streamlit run streamlit_app.py
# For Deployment: https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/llm-quickstart

import base64
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Check for debug mode from command line arguments or environment variable
# Note: Streamlit may consume --debug flag, so also check environment variable as fallback
# Usage: streamlit run streamlit_app.py --debug
#   OR:   CLUTCHAI_DEBUG=1 streamlit run streamlit_app.py
# Debug mode enables verbose terminal logging in ClutchAIAgent and other components
DEBUG_MODE = (
    "--debug" in sys.argv
    or "-debug" in sys.argv
    or os.environ.get("CLUTCHAI_DEBUG", "").lower() in ("1", "true", "yes")
)

# Logging will be set up after imports

# Add project root to Python path so we can import agents
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.resolve()

# Verify project root by checking for agents directory
if not (project_root / "agents").exists():
    # Try current working directory as fallback
    cwd_root = Path.cwd().resolve()
    if (cwd_root / "agents").exists():
        project_root = cwd_root
    else:
        raise FileNotFoundError(
            f"agents directory not found. Expected at: {project_root / 'agents'}"
        )

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import yaml
from logger import setup_logging, get_logger

# Setup logging early
setup_logging(debug=DEBUG_MODE)
logger = get_logger(__name__)

if DEBUG_MODE:
    logger.info("Debug mode enabled - verbose logging to terminal is active")

# Load Streamlit config to determine which agent to use
def load_streamlit_config() -> dict:
    """Load streamlit_config.yaml and return config dict."""
    config_path = project_root / "config" / "streamlit_config.yaml"

    if not config_path.exists():
        logger.info("streamlit_config.yaml not found. Using defaults.")
        return {"agent_type": "single", "max_history_messages": 20}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        agent_type = config.get("agent_type", "single")

        # Validate agent_type
        if agent_type not in ["single", "multi"]:
            logger.warning(
                f"Invalid agent_type '{agent_type}' in config. Defaulting to 'single'."
            )
            agent_type = "single"

        max_history_messages = config.get("max_history_messages", 20)
        # Ensure it's a positive integer
        try:
            max_history_messages = max(1, int(max_history_messages))
        except (ValueError, TypeError):
            logger.warning(f"Invalid max_history_messages value. Using default: 20")
            max_history_messages = 20

        logger.info(
            f"Loaded config: agent_type={agent_type}, max_history_messages={max_history_messages}"
        )
        return {
            "agent_type": agent_type,
            "max_history_messages": max_history_messages,
        }
    except Exception as e:
        logger.warning(f"Error loading streamlit_config.yaml: {e}. Using defaults.")
        return {"agent_type": "single", "max_history_messages": 20}


# Load Streamlit config
STREAMLIT_CONFIG = load_streamlit_config()
AGENT_TYPE = STREAMLIT_CONFIG["agent_type"]
MAX_HISTORY_MESSAGES = STREAMLIT_CONFIG["max_history_messages"]

# Import appropriate agent
if AGENT_TYPE == "multi":
    from agents.multi_agent import MultiAgentSystem

    AgentClass = MultiAgentSystem
    logger.info("Using Multi-Agent System")
else:
    from agents.single_agent import ClutchAIAgent

    AgentClass = ClutchAIAgent
    logger.info("Using Single Agent (ClutchAIAgent)")

# Load environment variables from .env file
env_file_location = project_root
env_file_path = env_file_location / ".env"

# Load .env file if it exists
if env_file_path.exists():
    load_dotenv(dotenv_path=env_file_path, override=True)
else:
    # Try loading from default location (current directory)
    load_dotenv(override=True)

GITHUB_URL = "https://github.com/mk-chang/ClutchAI"

# SUGGESTIONS for ClutchAI - fantasy basketball prompt suggestions (display label -> actual prompt)
# Format like demo-ai-assistant: :color[:material/icon:] Label
SUGGESTIONS = {
        ":red[:material/bar_chart:] Team performance summary": (
        "Summarize my team's performance and key stats. How am I doing?"
    ),
    ":blue[:material/play_arrow:] Who should I start this week?": (
        "Who should I start this week? Consider my roster, matchups, and recent performance."
    ),
    ":orange[:material/swap_horiz:] Trade advice": (
        "Give me trade advice for my roster. What players should I try to acquire or sell?"
    ),
    ":violet[:material/person_add:] Waiver wire pickups": (
        "Who are the best waiver wire pickups available in my league right now?"
    ),
}

# Set Streamlit page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="ClutchAI",
    page_icon="🏀",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.dialog("Legal disclaimer")
def show_disclaimer_dialog():
    st.caption("""
        This AI chatbot provides fantasy basketball advice based on available data.
        Advice may be inaccurate or outdated. Use your own judgment when making
        roster decisions. ClutchAI is not liable for any actions, losses, or
        damages resulting from the use of this chatbot.
    """)


def clear_conversation():
    st.session_state["messages"] = []
    st.session_state.initial_question = None
    st.session_state.selected_suggestion = None


# ChatGPT-style layout: user messages right-aligned, assistant left-aligned
st.markdown("""
<style>
  /* Header buttons (View source, Restart): top-right aligned, no border/box */
  div:has(.stLinkButton):has(.stButton) {
    display: flex !important;
    justify-content: flex-end !important;
    align-items: flex-start !important;
  }
  div:has(.stLinkButton):has(.stButton) .stLinkButton button,
  div:has(.stLinkButton):has(.stButton) .stButton > button {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
  }
  /* User avatar: no border/box */
  [data-testid="stChatMessageAvatar"][aria-label="user"] {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
  }
  /* User messages right-aligned */
  div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"][aria-label="user"]) {
    display: flex;
    margin-left: auto;
    max-width: 85%;
    flex-direction: row-reverse;
  }
  div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"][aria-label="user"]) > div {
    margin-left: auto;
    margin-right: 0;
  }
  /* Right-align text content in user messages */
  div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"][aria-label="user"]) [data-testid="stMarkdown"],
  div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"][aria-label="user"]) [data-testid="stText"] {
    text-align: right;
  }
</style>
""", unsafe_allow_html=True)

# Logo, title, and View source (like demo-ai-assistant)
logo_path = project_root / "assets" / "clutchai_logo.svg"
agent_avatar_path = project_root / "assets" / "clutchai_logo_negative.svg"
agent_avatar = None
if agent_avatar_path.exists():
    agent_avatar = f"data:image/svg+xml;base64,{base64.b64encode(agent_avatar_path.read_bytes()).decode('utf-8')}"
user_avatar = ":material/person:"

# Top bar: View source and Restart at very top right (first visible content)
_top_spacer, top_btn_col = st.columns([6, 1])
with top_btn_col:
    btn_row = st.container(horizontal=True)
    with btn_row:
        st.link_button("", url=GITHUB_URL, icon=":material/code:", type="tertiary", help="View source")
        st.button("", icon=":material/refresh:", on_click=clear_conversation, type="tertiary", help="Restart")

# Header like demo-ai-assistant: small icon/logo, then title row
if logo_path.exists():
    svg_content = logo_path.read_bytes()
    b64_svg = base64.b64encode(svg_content).decode("utf-8")
    st.markdown(
        f'<div style="font-size:2.5rem;line-height:1"><img src="data:image/svg+xml;base64,{b64_svg}" width="80" alt="ClutchAI Logo"/></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="font-size:2.5rem;line-height:1">🏀</div>',
        unsafe_allow_html=True,
    )
title_row = st.container(horizontal=True, vertical_alignment="top")
with title_row:
    st.title("ClutchAI", anchor=False, width="stretch")
st.caption(
    "Your AI fantasy basketball sports assistant, with real-time, context-aware insights on your Yahoo Fantasy Basketball Leagues."
)

# Credentials from env only (no sidebar)
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
yahoo_client_id = os.environ.get("YAHOO_CLIENT_ID", "")
yahoo_secret = os.environ.get("YAHOO_CLIENT_SECRET", "")
yahoo_league_id = os.environ.get("YAHOO_LEAGUE_ID", "58930")
team_name = os.environ.get("TEAM_NAME", "KATmandu Climbers")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialize agent in session state if not already initialized
if "agent" not in st.session_state:
    st.session_state["agent"] = None
if "agent_key" not in st.session_state:
    st.session_state["agent_key"] = None
if "initial_question" not in st.session_state:
    st.session_state.initial_question = None
if "selected_suggestion" not in st.session_state:
    st.session_state.selected_suggestion = None

# Demo-style: show suggestions before first question (like demo-ai-assistant)
user_just_asked_initial = (
    "initial_question" in st.session_state and st.session_state.initial_question
)
user_just_clicked_suggestion = (
    "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
)
user_first_interaction = user_just_asked_initial or user_just_clicked_suggestion
has_message_history = (
    "messages" in st.session_state and len(st.session_state.messages) > 0
)

# Show chat input, suggestions, and legal disclaimer first (visible during agent init)
if not user_first_interaction and not has_message_history:
    st.session_state["messages"] = []

    with st.container():
        st.chat_input("Ask a question...", key="initial_question")

        selected_suggestion = st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=list(SUGGESTIONS.keys()),
            key="selected_suggestion",
        )

    st.button(
        "&nbsp;:small[:gray[:material/balance: Legal disclaimer]]",
        type="tertiary",
        on_click=show_disclaimer_dialog,
    )

# Initialize or reinitialize agent on page load if credentials are available
if openai_api_key and yahoo_league_id:
    try:
        league_id_int = int(yahoo_league_id)
    except ValueError:
        st.error("Yahoo League ID must be a number.")
        st.stop()

    # Check if agent needs to be initialized or reinitialized
    agent_key = f"{AGENT_TYPE}_{openai_api_key[:10]}_{yahoo_client_id[:10] if yahoo_client_id else 'none'}_{league_id_int}_{team_name}_{DEBUG_MODE}"
    if st.session_state.get("agent_key") != agent_key or st.session_state["agent"] is None:
        agent_name = "Multi-Agent System" if AGENT_TYPE == "multi" else "ClutchAI Agent"
        with st.spinner(f"Initializing {agent_name}..."):
            try:
                logger.debug(
                    f"Initializing {agent_name} with debug mode: {DEBUG_MODE}, team_name: {team_name}"
                )
                init_kwargs = {
                    "yahoo_league_id": league_id_int,
                    "yahoo_client_id": yahoo_client_id or None,
                    "yahoo_client_secret": yahoo_secret or None,
                    "openai_api_key": openai_api_key,
                    "env_file_location": env_file_location,
                    "debug": DEBUG_MODE,
                }
                if AGENT_TYPE == "multi":
                    init_kwargs["team_name"] = team_name
                st.session_state["agent"] = AgentClass(**init_kwargs)
                st.session_state["agent_key"] = agent_key
                logger.debug(f"{agent_name} initialized successfully")
            except Exception as e:
                error_msg = f"Failed to initialize {agent_name}: {e}"
                st.error(error_msg)
                logger.error("Agent initialization error", exc_info=True)
                st.session_state["agent"] = None
                st.session_state["agent_key"] = None

# Stop here if user hasn't asked a question yet (after showing UI and running init)
if not user_first_interaction and not has_message_history:
    st.stop()

# Show chat input at bottom when a question has been asked (like demo-ai-assistant)
user_message = st.chat_input("Ask a follow-up...")

if not user_message:
    if user_just_asked_initial:
        user_message = st.session_state.initial_question
    if user_just_clicked_suggestion:
        user_message = SUGGESTIONS.get(st.session_state.selected_suggestion, "")

# Display chat messages from history (like demo-ai-assistant)
for i, msg in enumerate(st.session_state.messages):
    avatar = user_avatar if msg["role"] == "user" else agent_avatar
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant":
            st.container()  # Fix ghost message bug
        st.markdown(msg["content"])

# Process user message
if user_message:
    # Fix LaTeX interpretation
    user_message = user_message.replace("$", r"\$")

    # Check credentials before processing
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    if not yahoo_league_id:
        st.info("Please add your Yahoo League ID to continue.")
        st.stop()
    if st.session_state["agent"] is None:
        st.error("Agent not initialized. Please check your credentials.")
        st.stop()

    # Add user message to chat (like demo-ai-assistant)
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user", avatar=user_avatar):
        st.text(user_message)

    # Get agent response (like demo-ai-assistant)
    with st.chat_message("assistant", avatar=agent_avatar):
        with st.spinner("Thinking..."):
            try:
                all_history = st.session_state.messages[:-1]
                logger.debug(
                    f"Processing prompt: {user_message[:100]}... "
                    f"(using {len(all_history)} history messages)"
                )
                response = st.session_state["agent"].chat(
                    user_message,
                    conversation_history=all_history,
                    max_history_messages=MAX_HISTORY_MESSAGES,
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                with st.container():
                    st.markdown(response)
                logger.debug(f"Response generated ({len(response)} characters)")
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                logger.error("Chat error", exc_info=True)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
                with st.container():
                    st.markdown(error_msg)
