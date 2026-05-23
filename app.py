import streamlit as st
from src.rag import ask
import os
from dotenv import load_dotenv

# Set page configuration
st.set_page_config(
    page_title="Feedback RAG",
    page_icon="🛒",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Bridge Streamlit Cloud secrets to environment variables for local/cloud compatibility
if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
    os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# --- Sidebar Configuration ---
st.sidebar.title("🛒 Project Info")
st.sidebar.info(
    """
    - **Dataset**: 850 Amazon Fine Food Reviews
    - **Embeddings**: `gemini-embedding-001`
    - **Generation**: `Gemma-3-27B-IT`
    """
)

st.sidebar.divider()

st.sidebar.subheader("Example Questions")
examples = [
    "What are common complaints about packaging?",
    "Do dogs like the taste of these treats?",
    "Is this product gluten-free?",
    "How does this brand compare to Starbucks?"
]

# Use session state to handle input value from examples
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""

def set_query(q):
    st.session_state.user_query = q

for q in examples:
    if st.sidebar.button(q):
        set_query(q)

st.sidebar.divider()
st.sidebar.markdown("[View on GitHub](https://github.com/Eternity2401/feedback-rag)")

# --- Main Area ---
st.title("🛒 Amazon Reviews RAG")
st.markdown("Ask anything about the product reviews in our database. The AI will answer based ONLY on retrieved reviews.")

st.divider()

# Input area
query = st.text_input(
    "Enter your question:", 
    value=st.session_state.user_query,
    placeholder="e.g., Are the chips usually crushed upon arrival?"
)

ask_button = st.button("Ask AI")

if ask_button or (query and query != st.session_state.user_query):
    if not query:
        st.warning("Please enter a question first.")
    else:
        # Update session state
        st.session_state.user_query = query
        
        with st.spinner("Analyzing reviews and generating answer..."):
            try:
                result = ask(query)
                
                # Display Answer
                st.subheader("Answer")
                st.success(result['answer'])
                
                st.divider()
                
                # Display Sources
                with st.expander("View 5 Source Reviews"):
                    for i, source in enumerate(result['sources'], 1):
                        meta = source['metadata']
                        st.markdown(f"**[{i}] {meta.get('Summary', 'No Summary')}**")
                        st.markdown(f"*Rating: {meta.get('Score', 'N/A')}/5*")
                        st.text_area(f"Full text for Review {i}", value=meta.get('Text', ''), height=100, disabled=True)
                        st.divider()
                        
            except Exception as e:
                err_msg = str(e).lower()
                if any(kw in err_msg for kw in ["429", "resource_exhausted", "quota"]):
                    st.error("Rate limit hit (Google Gemini Free Tier). Please wait 60 seconds and try again.")
                else:
                    st.error(f"An unexpected error occurred: {e}")

st.divider()
st.caption("Powered by Google Gemini & ChromaDB. Built with Streamlit.")
