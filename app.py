import streamlit as st
import faiss
import numpy as np
import pickle
import os
import time
import re
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Offline RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "models/phi-2.gguf"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "chunk_metadata.pkl"
TOP_K = 3

# -----------------------------------------------------------------------------
# Custom Styling (CSS) - Hyper-Clean Professional
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    /* Global Settings */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #FFFFFF;
        color: #111111;
        -webkit-font-smoothing: antialiased;
    }
    
    /* App Container */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Header Typography */
    h1 {
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #111111;
    }
    
    h3 {
        font-weight: 500;
        color: #666666;
    }
    
    /* Sidebar - Minimal & Light */
    [data-testid="stSidebar"] {
        background-color: #FAFAFA;
        border-right: 1px solid #F0F0F0;
    }
    
    /* Chat Input - Premium Search Bar Feel */
    .stChatInputContainer {
        padding-bottom: 3rem;
        max-width: 768px; /* Standard reading width */
        margin: 0 auto;
    }
    
    div[data-testid="stChatInput"] {
        border-radius: 16px; /* Smooth rounded corners */
        border: 1px solid #E5E5E5;
        background-color: #FFFFFF;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04); /* Soft, expensive shadow */
        color: #111111;
        transition: border 0.2s ease, box-shadow 0.2s ease;
    }
    
    div[data-testid="stChatInput"]:focus-within {
        border-color: #D1D1D1;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.06);
    }
    
    /* Chat Bubbles - Clean & Distraction Free */
    .stChatMessage {
        background-color: transparent;
        border: none;
        padding: 1.5rem 0;
        max-width: 768px;
        margin: 0 auto;
    }
    
    /* Message Text */
    .stChatMessage div {
        line-height: 1.6;
        font-size: 16px;
    }
    
    /* Avatars - Minimal squares/circles */
    .stChatMessage .stChatMessageAvatar {
        background-color: #F5F5F5;
        color: #333;
        border-radius: 6px; /* Slightly squared for tech feel */
    }
    
    /* Welcome Screen - Centered & Typgoraphic */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 60vh;
        text-align: center;
    }
    
    .welcome-title {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #111111;
        letter-spacing: -0.03em;
    }
    
    .welcome-subtitle {
        font-size: 1.1rem;
        color: #888888;
        font-weight: 400;
    }

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Resource Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    """Loads models and index once."""
    resources = {}
    
    try:
        resources['embedder'] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        try:
            resources['index'] = faiss.read_index(INDEX_FILE)
            with open(METADATA_FILE, 'rb') as f:
                resources['chunks'] = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading index: {e}")
            return None
    else:
        # Silent fail for cleaner UI, no ugly warning banners
        pass

    if os.path.exists(MODEL_PATH):
        try:
            resources['llm'] = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=True, n_threads=6) 
        except Exception as e:
            st.error(f"Error loading LLM: {e}")
            return None
    
    return resources

def retrieve(query, resources):
    if not resources or 'index' not in resources: return []
    index = resources['index']
    embedder = resources['embedder']
    chunks = resources['chunks']
    
    query_vector = embedder.encode([query])
    query_vector = np.array(query_vector).astype('float32')
    distances, indices = index.search(query_vector, TOP_K)
    
    results = []
    for i in indices[0]:
        if i < len(chunks):
            content = chunks[i].page_content
            # Clean up text
            content = content.replace('\n', ' ')
            content = re.sub(r'<[^>]+>', '', content) # Remove HTML tags
            results.append(content)
    return results

# -----------------------------------------------------------------------------
# Application Layout
# -----------------------------------------------------------------------------
resources = load_resources()

# Sidebar - Minimal
with st.sidebar:
    st.markdown("### **Offline RAG**")
    st.markdown('<div style="font-size: 12px; color: #888; margin-bottom: 20px;">v1.0 • Local Execution</div>', unsafe_allow_html=True)
    
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome Screen (No Icon, Just Clean Text)
if len(st.session_state.messages) == 0:
    st.markdown("""
        <div class="welcome-container">
            <div class="welcome-title">How can I help?</div>
            <div class="welcome-subtitle">Ask anything about your documents.</div>
        </div>
        """, unsafe_allow_html=True)
else:
    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Input Query
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate
    if resources and 'llm' in resources:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            context_chunks = retrieve(prompt, resources)
            context_text = "\n\n".join(context_chunks)
            system_prompt = f"Instruct: Answer strictly from context.\nContext:\n{context_text}\nQuestion:\n{prompt}\nOutput:\n"
            
            stream = resources['llm'](system_prompt, max_tokens=256, stop=["Instruct:", "Question:"], stream=True)
            
            try:
                for output in stream:
                    token = output['choices'][0]['text']
                    full_response += token
                    response_placeholder.write(full_response + "▌")
                response_placeholder.write(full_response)
            except:
                pass
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.caption("System Offline.")
