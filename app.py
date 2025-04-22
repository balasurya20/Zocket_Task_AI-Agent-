import streamlit as st
import time
import pandas as pd
import os
from agent_manager import AgentManager

st.set_page_config(
    page_title="Web Content Analyzer",
    page_icon="🔍",
    layout="wide",
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .results-container {
        padding: 20px;
        border-radius: 5px;
        background-color: #f9f9f9;
        margin-top: 20px;
    }
    .sentiment-positive {
        color: green;
        font-weight: bold;
    }
    .sentiment-negative {
        color: red;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #888;
        font-weight: bold;
    }
    .keywords-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .keyword-tag {
        background-color: #e1ecf4;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 14px;
        color: #39739d;
    }
    .raw-content {
        max-height: 300px;
        overflow-y: auto;
        padding: 10px;
        background-color: #f5f5f5;
        border-radius: 5px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

def get_agent_manager():
    if 'hf_api_key' not in st.session_state:
        st.session_state.hf_api_key = os.environ.get("HF_API_KEY", "")
    if ('agent_manager' not in st.session_state or 
            st.session_state.get('current_api_key', '') != st.session_state.hf_api_key):
        st.session_state.agent_manager = AgentManager(hf_api_key=st.session_state.hf_api_key)
        st.session_state.current_api_key = st.session_state.hf_api_key
    return st.session_state.agent_manager

if 'results' not in st.session_state:
    st.session_state.results = None

if 'is_loading' not in st.session_state:
    st.session_state.is_loading = False

def process_url():
    url = st.session_state.url_input
    if not url:
        st.error("Please enter a URL.")
        return
    st.session_state.is_loading = True
    st.session_state.results = None
    agent_manager = get_agent_manager()
    result = agent_manager.process_website(url)
    st.session_state.results = result
    st.session_state.is_loading = False

st.title("🔍 Web Content Analyzer")
st.subheader("Extract insights from any webpage using AI agents")

with st.sidebar:
    st.subheader("Configuration")
    api_key = st.text_input(
        "Hugging Face API Key (optional)", 
        value=st.session_state.get('hf_api_key', ''),
        type="password",
        help="Enter your Hugging Face API key. The app will work without it, but may be rate-limited."
    )
    if api_key != st.session_state.get('hf_api_key', ''):
        st.session_state.hf_api_key = api_key
        if 'agent_manager' in st.session_state:
            del st.session_state.agent_manager
    st.markdown("""
    ### About
    This application uses agentic AI to analyze web content. It extracts the main text, summarizes it, analyzes sentiment, and identifies key topics.
    
    ### Models Used
    - **Summarization**: facebook/bart-large-cnn
    - **Sentiment Analysis**: distilbert-base-uncased-finetuned-sst-2-english
    - **Text Generation**: mistralai/Mixtral-8x7B-Instruct-v0.1
    
    ### How to Get a Hugging Face API Key
    1. Create a free account at [huggingface.co](https://huggingface.co)
    2. Go to your profile settings
    3. Generate a new API key
    4. Paste it here
    """)

st.text_input("Enter website URL:", key="url_input", 
              placeholder="https://example.com", 
              help="Enter the full URL including http:// or https://")

st.button("Analyze Website", on_click=process_url, key="process_button")

if st.session_state.is_loading:
    with st.spinner("Analyzing website content... This may take a minute."):
        time.sleep(0.5)

if st.session_state.results:
    results = st.session_state.results
    if not results["success"]:
        st.error(f"Error: {results.get('error', 'An unknown error occurred.')}")
    else:
        st.markdown("## Analysis Results")
        tab1, tab2, tab3 = st.tabs(["Summary", "Keywords & Sentiment", "Raw Content"])
        with tab1:
            st.markdown(f"### {results['title']}")
            st.markdown("#### Summary")
            st.markdown(results['summary'])
        with tab2:
            st.markdown("### Keywords")
            cols = st.columns(5)
            for i, keyword in enumerate(results['keywords']):
                with cols[i % 5]:
                    st.markdown(f"""
                        <div class="keyword-tag">{keyword}</div>
                    """, unsafe_allow_html=True)
            st.markdown("### Sentiment Analysis")
            sentiment_text = results['sentiment']
            sentiment_class = "sentiment-neutral"
            if "POSITIVE" in sentiment_text:
                sentiment_class = "sentiment-positive"
            elif "NEGATIVE" in sentiment_text:
                sentiment_class = "sentiment-negative"
            st.markdown(f"""
                <p class="{sentiment_class}">{sentiment_text}</p>
            """, unsafe_allow_html=True)
            if "POSITIVE" in sentiment_text:
                st.markdown("The content has an overall positive tone.")
            elif "NEGATIVE" in sentiment_text:
                st.markdown("The content has an overall negative tone.")
            else:
                st.markdown("The content has a neutral tone.")
        with tab3:
            st.markdown("### Raw Content")
            st.markdown("""
                <div class="raw-content">
                    {content}
                </div>
            """.format(content=results['raw_content'].replace('\n', '<br>')), unsafe_allow_html=True)

st.markdown("---")
st.markdown("Built with Hugging Face Inference API and agentic AI techniques")
