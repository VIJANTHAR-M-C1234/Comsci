import streamlit as st
import sys
import os

# Add the project root to sys.path so we can import backend packages correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.retriever import get_retriever, get_connection_info
from backend.chatbot import ask_chatbot, transcribe_audio, SUPPORTED_LANGUAGES

# Page configuration
st.set_page_config(
    page_title="NCERT AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a Clean, Minimalist (ChatGPT-like) Design
chatgpt_css = """
<style>
/* Base Theme */
.stApp {
    background-color: #343541; /* ChatGPT dark gray */
    color: #ECECF1;
    font-family: 'Söhne', 'Inter', sans-serif;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #202123 !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: #ECECF1;
}

/* Hide original streamlit elements for clean UI */
header {visibility: hidden;}
footer {visibility: hidden;}

/* User Chat Message */
.stChatMessage:nth-child(odd) {
    background-color: #343541;
}

/* AI Chat Message */
.stChatMessage:nth-child(even) {
    background-color: #444654;
    border-top: 1px solid rgba(32,33,35,0.5);
    border-bottom: 1px solid rgba(32,33,35,0.5);
}

/* Chat Input Styling */
.stChatInputContainer {
    background-color: #40414F !important;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    box-shadow: 0 0 15px rgba(0,0,0,0.1) !important;
}

/* Metadata pills (small tags for RAG context) */
.meta-pill {
    display: inline-block;
    background: #202123;
    border: 1px solid #565869;
    border-radius: 8px;
    padding: 2px 8px;
    font-size: 0.75rem;
    color: #C5C5D2;
    margin-right: 5px;
    margin-bottom: 5px;
}

/* Language pill - highlighted in accent color */
.lang-pill {
    display: inline-block;
    background: #1a3a4a;
    border: 1px solid #19c37d;
    border-radius: 8px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: #19c37d;
    margin-right: 5px;
    margin-bottom: 5px;
    font-weight: 600;
}

/* Main title styling */
h1 {
    text-align: center;
    font-weight: 600;
    margin-top: -40px;
    padding-bottom: 20px;
}

/* Language selector label styling */
.language-label {
    font-size: 0.85rem;
    color: #C5C5D2;
    margin-bottom: 4px;
}

/* ── Force sidebar to always stay open ─────────────────────────────────────── */
/* Hide the collapse/expand arrow button completely */
[data-testid="collapsedControl"] {
    display: none !important;
}

/* Ensure the sidebar never slides off-screen */
section[data-testid="stSidebar"] {
    display: flex !important;
    transform: none !important;
    min-width: 260px !important;
    width: 300px !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* Prevent the toggle hamburger button in the header from hiding the sidebar */
[data-testid="stSidebarNav"] {
    display: block !important;
}
</style>
"""
st.markdown(chatgpt_css, unsafe_allow_html=True)

# Initialize chat history early so sidebar can read it safely
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?", "metadata": {}}
    ]

# Cache connection info once so it never disappears during re-renders
if "conn_info" not in st.session_state:
    st.session_state.conn_info = get_connection_info()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("## 🤖 NCERT AI Assistant")
    st.markdown("A highly intelligent RAG tutor model.")

    st.divider()

    # ── Vector DB Connection Status — always at top, always visible ───────────
    conn = st.session_state.conn_info
    db_color = "#19c37d" if conn["is_cloud"] else "#f5a623"
    st.markdown(
        f"""
        <div style="
            background:#1a1a2e;
            border:1px solid {db_color};
            border-radius:10px;
            padding:10px 14px;
            margin-bottom:4px;
        ">
          <div style="font-size:0.75rem;color:#8E8EA0;margin-bottom:3px;letter-spacing:0.04em;">🗄️ VECTOR DATABASE</div>
          <div style="font-size:0.95rem;font-weight:700;color:{db_color};">{conn['mode']}</div>
          <div style="font-size:0.7rem;color:#8E8EA0;margin-top:4px;word-break:break-all;line-height:1.4;">{conn['detail']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### ⚙️ Settings")
    difficulty = st.selectbox(
        "Explanation Difficulty:",
        ["Beginner", "Detailed"],
        help="Adjust the depth of the explanation."
    )

    st.markdown("<div class='language-label'>🌐 Answer Language</div>", unsafe_allow_html=True)
    answer_language = st.selectbox(
        "Answer Language",
        SUPPORTED_LANGUAGES,
        index=0,
        help="Choose the language in which the AI will respond. The model always retrieves from the English vector DB, but answers in your chosen language.",
        label_visibility="collapsed"
    )
    st.markdown(
        f"<small style='color:#8E8EA0;'>Model will reply in <b style='color:#19c37d;'>{answer_language}</b></small>",
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("#### 🎙️ Voice Input")
    st.markdown("<small style='color: #8E8EA0;'>Optional: Ask with audio</small>", unsafe_allow_html=True)
    audio_val = st.audio_input("Record Voice", label_visibility="collapsed")

    st.divider()
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I help you today?", "metadata": {}}
        ]
        st.rerun()


# ----------------- MAIN CHAT AREA -----------------

# Header
st.title("NCERT AI Assistant")

# Initialize retriever
if "retriever" not in st.session_state:
    try:
        retriever, db_mode = get_retriever()
        st.session_state.retriever = retriever
        st.session_state.db_mode = db_mode
    except Exception as e:
        st.warning(f"⚠️ Vector Database is not ready: {e}")
        st.session_state.retriever = None
        st.session_state.db_mode = "unavailable"

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("metadata"):
            meta = message["metadata"]
            subject = meta.get("subject", "General")
            lang_detected = meta.get("language", "")
            answer_lang = meta.get("answer_language", "English")
            translated = meta.get("translated_query", "")
            
            pills_html = f'<span class="meta-pill">📚 {subject}</span>'
            if lang_detected:
                pills_html += f'<span class="meta-pill">🗣️ Detected: {lang_detected}</span>'
            pills_html += f'<span class="lang-pill">🌐 Reply: {answer_lang}</span>'
            if translated:
                pills_html += f'<span class="meta-pill">🔍 {translated}</span>'
            st.markdown(pills_html, unsafe_allow_html=True)
        
        st.markdown(message["content"])
                    
# Determine prompt from Audio OR Text Input
prompt_text = None

if audio_val:
    with st.spinner("Transcribing audio..."):
        transcription = transcribe_audio(audio_val.getvalue())
        if transcription and "Error" not in transcription:
            prompt_text = transcription
        else:
            st.error(f"Voice Recognition Failed: {transcription}")

typed_prompt = st.chat_input("Message NCERT AI...")
if typed_prompt:
    prompt_text = typed_prompt

# Process prompt
if prompt_text:
    # Save & Print User Prompt
    st.session_state.messages.append({"role": "user", "content": prompt_text, "metadata": {}})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Assistant Response
    with st.chat_message("assistant"):
        if st.session_state.retriever is None:
            response_text = "Error: Vector database not initialized."
            st.error(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text, "metadata": {}})
        else:
            with st.spinner(f"Synthesizing answer in {answer_language}..."):
                result = ask_chatbot(
                    prompt_text,
                    st.session_state.retriever,
                    st.session_state.messages,
                    difficulty,
                    answer_language=answer_language
                )
                response_text = result["response"]
                metadata = result["metadata"]
                
                # Show metadata pills
                if metadata:
                    subject = metadata.get("subject", "General")
                    lang_detected = metadata.get("language", "")
                    answer_lang = metadata.get("answer_language", "English")
                    translated = metadata.get("translated_query", "")
                    
                    pills_html = f'<span class="meta-pill">📚 {subject}</span>'
                    if lang_detected:
                        pills_html += f'<span class="meta-pill">🗣️ Detected: {lang_detected}</span>'
                    pills_html += f'<span class="lang-pill">🌐 Reply: {answer_lang}</span>'
                    if translated:
                        pills_html += f'<span class="meta-pill">🔍 {translated}</span>'
                    st.markdown(pills_html, unsafe_allow_html=True)
                
                st.markdown(response_text)
                
    # Store assistant response in history
    st.session_state.messages.append({"role": "assistant", "content": response_text, "metadata": metadata})
