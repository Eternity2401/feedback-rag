import streamlit as st
from src.rag import ask
import os
import json
import time
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

from agent import retrieve as agent_retrieve
from agent import classify as agent_classify
from agent import route as agent_route

# Set page configuration
st.set_page_config(
    page_title="Feedback RAG & MCP Agent",
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
    - **Protocol**: `Model Context Protocol (v2)`
    - **Deployment**: Streamlit Cloud + Render Free
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
    if st.sidebar.button(q, key=f"side_{q}"):
        set_query(q)

st.sidebar.divider()
st.sidebar.markdown("[View on GitHub](https://github.com/Eternity2401/feedback-rag)")

# --- Main Area ---
st.title("🛒 Amazon Reviews RAG & Autonomous Agent")

# Create three tabs
tab1, tab2, tab3 = st.tabs(["🛒 Ask Reviews", "🤖 Triage Agent", "🔌 MCP Server & Protocol"])

# --- Tab 1: Ask Reviews (Existing RAG Q&A - 100% Intact) ---
with tab1:
    st.header("🛒 Amazon Reviews Q&A")
    st.markdown("Ask anything about the product reviews in our database. The AI will answer based ONLY on retrieved reviews.")
    
    st.divider()

    # Input area
    query = st.text_input(
        "Enter your question:", 
        value=st.session_state.user_query,
        placeholder="e.g., Are the chips usually crushed upon arrival?"
    )

    ask_button = st.button("Ask AI", key="tab1_ask_btn")

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
                            st.text_area(f"Full text for Review {i}", value=meta.get('Text', ''), height=100, disabled=True, key=f"review_{i}")
                            st.divider()
                            
                except Exception as e:
                    err_msg = str(e).lower()
                    if any(kw in err_msg for kw in ["429", "resource_exhausted", "quota"]):
                        st.error("Rate limit hit (Google Gemini Free Tier). Please wait 60 seconds and try again.")
                    else:
                        st.error(f"An unexpected error occurred: {e}")

# --- Tab 2: Triage Agent (Existing Agent - 100% Intact) ---
with tab2:
    st.header("🤖 Customer Feedback Triage Agent")
    
    # 1. SECTION: Triage New Feedback
    st.subheader("Triage New Feedback")
    st.markdown("Paste a customer feedback below. The agent will retrieve similar past entries, classify it, and route it to the right team.")
    
    feedback_text = st.text_area(
        "Customer feedback",
        placeholder="e.g., App crashes every time I open it on Android 14. Lost all my data after the latest update. This is unacceptable for a paid app.",
        key="tab2_feedback_input"
    )
    
    triage_button = st.button("Triage this feedback", key="tab2_triage_btn")
    
    if triage_button:
        if not feedback_text.strip():
            st.warning("Please enter some customer feedback first.")
        else:
            with st.spinner("Retrieving similar entries and classifying... (~20 seconds due to free-tier rate limits)"):
                try:
                    # Retrieve 3 similar examples
                    similar_examples = agent_retrieve.retrieve_similar(feedback_text, k=3)
                    
                    # Classify
                    classification = agent_classify.classify(feedback_text, similar_examples)
                    
                    # Route
                    routing = agent_route.route(classification)
                    
                    # Display results:
                    st.success("Triage Completed successfully!")
                    
                    # Sentiment, Category, Urgency Badges in a single row
                    sent = classification.get("sentiment", "neutral").lower()
                    sent_colors = {
                        "positive": "#28a745", # green
                        "negative": "#dc3545", # red
                        "neutral": "#6c757d",  # gray
                        "mixed": "#fd7e14"     # orange
                    }
                    sent_color = sent_colors.get(sent, "#6c757d")
                    
                    cat = classification.get("category", "other").lower()
                    
                    urg = classification.get("urgency", "low").lower()
                    urg_colors = {
                        "critical": "#dc3545", # red
                        "high": "#fd7e14",     # orange
                        "medium": "#ffc107",   # yellow
                        "low": "#28a745"       # green
                    }
                    urg_color = urg_colors.get(urg, "#28a745")
                    
                    badge_html = f"""
                    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">
                        <span style="background-color: {sent_color}; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px;">Sentiment: {sent.capitalize()}</span>
                        <span style="background-color: #f1f3f5; color: #495057; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px; border: 1px solid #dee2e6;">Category: {cat.capitalize()}</span>
                        <span style="background-color: {urg_color}; color: white; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px;">Urgency: {urg.capitalize()}</span>
                    </div>
                    """
                    st.markdown(badge_html, unsafe_allow_html=True)
                    
                    # Routing Info box
                    st.info(
                        f"**Routed to**: `{routing.get('destination', 'support').upper()}` "
                        f"({routing.get('queue', 'normal')}) — **Priority Score**: `{routing.get('priority_score', 3)}/10`"
                    )
                    
                    # Reasoning expander
                    with st.expander("Reasoning"):
                        st.write(classification.get("reasoning", "No reasoning provided."))
                        
                    # Similar reviews expander
                    with st.expander("Top 3 similar past feedbacks (retrieved from ChromaDB)"):
                        for i, ex in enumerate(similar_examples, 1):
                            st.markdown(f"**[{i}] {ex.get('source', 'amazon').upper()} Review** (Cosine Distance: `{ex.get('distance', 0.0):.4f}`)")
                            st.write(ex.get('text', 'No text.'))
                            st.divider()
                            
                except Exception as e:
                    st.error("Triage failed. This is usually a free-tier rate limit. Please wait 60 seconds and try again.")
                    print(f"[ERROR] Triage failed: {e}")

    # 2. SECTION: Last Batch Run Results
    st.divider()
    st.subheader("Last Batch Run Results")
    st.markdown("Results from the most recent `python -m agent.run` execution on `data/triage_feedbacks.csv`.")
    
    results_csv = Path("agent/outputs/triaged_results.csv")
    if results_csv.exists():
        try:
            df = pd.read_csv(results_csv)
            
            # Metrics Row
            total_cnt = len(df)
            eng_cnt = len(df[df["destination"] == "engineering"])
            sup_cnt = len(df[df["destination"] == "support"])
            
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Total Feedbacks", total_cnt)
            m_col2.metric("Routed to Engineering", eng_cnt)
            m_col3.metric("Routed to Support", sup_cnt)
            
            # Filters side-by-side
            categories = ["All"] + sorted(list(df["category"].dropna().unique()))
            urgencies = ["All"] + sorted(list(df["urgency"].dropna().unique()))
            destinations = ["All"] + sorted(list(df["destination"].dropna().unique()))
            
            f_col1, f_col2, f_col3 = st.columns(3)
            sel_cat = f_col1.selectbox("Filter by Category", categories, key="f_cat")
            sel_urg = f_col2.selectbox("Filter by Urgency", urgencies, key="f_urg")
            sel_dest = f_col3.selectbox("Filter by Destination", destinations, key="f_dest")
            
            # Apply filters
            filtered_df = df.copy()
            if sel_cat != "All":
                filtered_df = filtered_df[filtered_df["category"] == sel_cat]
            if sel_urg != "All":
                filtered_df = filtered_df[filtered_df["urgency"] == sel_urg]
            if sel_dest != "All":
                filtered_df = filtered_df[filtered_df["destination"] == sel_dest]
                
            # Truncated text display
            display_df = filtered_df.copy()
            if "text" in display_df.columns:
                display_df["text"] = display_df["text"].apply(lambda t: str(t)[:80] + "..." if len(str(t)) > 80 else str(t))
                
            cols_to_show = ["id", "text", "sentiment", "category", "urgency", "destination", "queue", "priority_score"]
            cols_to_show = [c for c in cols_to_show if c in display_df.columns]
            
            st.dataframe(display_df[cols_to_show], use_container_width=True, hide_index=True)
            
            # Expander for full reasoning
            with st.expander("Full reasoning per feedback"):
                for _, row in filtered_df.iterrows():
                    st.markdown(f"**ID {row['id']}** | *Category*: `{row['category']}` | *Urgency*: `{row['urgency']}`")
                    st.write(f"**Reasoning**: {row.get('reasoning', 'No reasoning.')}")
                    st.divider()
                    
        except Exception as e:
            st.error(f"Error loading batch results: {e}")
    else:
        st.info("No batch results yet. Run `python -m agent.run` from the project root to generate results.")


# --- Tab 3: MCP Server & Protocol (NEW Feature) ---
with tab3:
    st.header("🔌 Model Context Protocol (MCP) Integration")
    st.markdown(
        "Standardized interoperability layer exposing **Feedback RAG** and **Autonomous Triage** "
        "capabilities to external AI clients (Claude Desktop, Cursor IDE, Windsurf, LangChain, Multi-Agent Swarms) via MCP v2."
    )
    
    st.divider()

    # Section A: Live Endpoint Status
    render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME", "feedback-rag-mcp.onrender.com")
    public_mcp_url = f"https://{render_host}/mcp"
    health_url = f"https://{render_host}/healthz"

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("MCP Protocol", "v2 (Official SDK)")
    col_stat2.metric("Transports", "Streamable HTTP + stdio")
    col_stat3.metric("Capabilities", "4 Tools | 2 Res | 1 Prompt")

    st.markdown("#### 🌐 Public Deployment Endpoints")
    st.code(f"Streamable HTTP MCP Endpoint: {public_mcp_url}\nHealth Check Endpoint:       {health_url}", language="text")

    st.caption("⚡ Note: Render free tier services automatically spin down after 15 minutes of inactivity; initial connection may experience a 30s cold-start.")

    st.divider()

    # Section B: Architecture Flowchart
    st.subheader("🏛️ Architecture Overview")
    st.markdown(
        """
        ```text
                         EXTERNAL CLIENT LAYER
             (Claude Desktop / Cursor IDE / Remote AI Agents)
                                  │
                                  │ JSON-RPC (Streamable HTTP / stdio)
                                  ▼
                    ┌───────────────────────────┐
                    │    FEEDBACK-RAG MCP       │
                    │   SERVER (FastMCP v2)     │
                    │  - Rate Limiter & Cache   │
                    │  - Health Check /healthz  │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
          ┌───────────────────┐       ┌───────────────────┐
          │   CORE RAG ENGINE │       │   TRIAGE AGENT    │
          │  - ChromaDB (850) │       │  - Gemma-3-27B    │
          │  - Gemini Embed   │       │  - Deterministic  │
          └───────────────────┘       └───────────────────┘
        ```
        """
    )

    st.divider()

    # Section C: Exposed Tools & Resources Catalog
    st.subheader("🧰 Exposed MCP Tools & Resources")
    
    col_cat1, col_cat2 = st.columns(2)
    with col_cat1:
        st.markdown("##### 🛠️ Tools")
        st.markdown(
            """
            - **`query_reviews_rag(question, top_k)`**: Natural-language RAG answering with review citations.
            - **`semantic_search_reviews(query, k)`**: Vector similarity search returning nearest review neighbors & distance scores.
            - **`triage_customer_feedback(feedback_text)`**: End-to-end feedback classification and queue routing.
            - **`get_triage_batch_metrics()`**: Read-only analytical summary of batch triage distribution (0 API cost).
            """
        )

    with col_cat2:
        st.markdown("##### 📦 Resources & Prompts")
        st.markdown(
            """
            - **`reviews://dataset-summary`**: Read-only ChromaDB collection metadata (850 items, 3072 dims).
            - **`triage://queues`**: Read-only queue schema (P0, P1, P2, refunds, backlog, wins_board, priority).
            - **`feedback_triage_prompt`**: Standardized prompt template for feedback triage.
            """
        )

    st.divider()

    # Section D: Interactive MCP Protocol Tool Tester
    st.subheader("🧪 Interactive MCP Protocol Tool Tester")
    st.markdown("Execute MCP tools through the standardized MCP server layer to inspect real JSON-RPC response payloads.")

    tool_choice = st.selectbox(
        "Select MCP Tool to Execute:",
        [
            "query_reviews_rag",
            "semantic_search_reviews",
            "triage_customer_feedback",
            "get_triage_batch_metrics"
        ],
        key="mcp_tool_choice"
    )

    # Dynamic inputs based on selected tool
    tool_input_kwargs = {}
    if tool_choice == "query_reviews_rag":
        q_in = st.text_input("question", value="What are common complaints about packaging?", key="mcp_q_in")
        k_in = st.slider("top_k", min_value=1, max_value=10, value=5, key="mcp_k_in")
        tool_input_kwargs = {"question": q_in, "top_k": k_in}

    elif tool_choice == "semantic_search_reviews":
        s_in = st.text_input("query", value="crushed chips and broken boxes", key="mcp_s_in")
        k_s_in = st.slider("k", min_value=1, max_value=10, value=3, key="mcp_ks_in")
        tool_input_kwargs = {"query": s_in, "k": k_s_in}

    elif tool_choice == "triage_customer_feedback":
        fb_in = st.text_area("feedback_text", value="Payment failed thrice in checkout but money was deducted three times. Need refund ASAP.", key="mcp_fb_in")
        tool_input_kwargs = {"feedback_text": fb_in}

    elif tool_choice == "get_triage_batch_metrics":
        st.caption("This tool is read-only and requires no input parameters.")
        tool_input_kwargs = {}

    run_mcp_btn = st.button("🚀 Execute MCP Tool Call", key="run_mcp_call_btn")

    if run_mcp_btn:
        with st.spinner(f"Executing MCP tool '{tool_choice}' via MCP Server..."):
            try:
                start_t = time.time()
                
                # Execute through the MCP Server layer
                from mcp_server import server
                if tool_choice == "query_reviews_rag":
                    tool_res = server.query_reviews_rag(**tool_input_kwargs)
                elif tool_choice == "semantic_search_reviews":
                    tool_res = server.semantic_search_reviews(**tool_input_kwargs)
                elif tool_choice == "triage_customer_feedback":
                    tool_res = server.triage_customer_feedback(**tool_input_kwargs)
                elif tool_choice == "get_triage_batch_metrics":
                    tool_res = server.get_triage_batch_metrics()
                else:
                    tool_res = {"error": "Unknown tool"}

                elapsed = time.time() - start_t

                st.success(f"✅ Tool Call Completed in {elapsed:.2f}s | Status: Success")
                st.markdown("**MCP Request / Response Payload (JSON-RPC Wire Format):**")
                st.json(tool_res)

            except Exception as e:
                st.error(f"MCP Tool execution failed: {e}")

    st.divider()

    # Section E: Client Configuration Snippets
    st.subheader("💻 Client Configuration Snippets")
    st.markdown("Easily connect **Claude Desktop** or **Cursor IDE** to this MCP Server:")

    tab_conf1, tab_conf2 = st.tabs(["Claude Desktop (Local stdio)", "Cursor IDE / Remote HTTP"])
    
    with tab_conf1:
        st.markdown("Add this to your `claude_desktop_config.json`:")
        claude_cfg = {
            "mcpServers": {
                "feedback-rag": {
                    "command": "python",
                    "args": ["-m", "mcp_server.server"],
                    "cwd": "C:/path/to/feedback-rag"
                }
            }
        }
        st.code(json.dumps(claude_cfg, indent=2), language="json")

    with tab_conf2:
        st.markdown("Add this to your Cursor `.cursor/mcp.json`:")
        cursor_cfg = {
            "mcpServers": {
                "feedback-rag": {
                    "url": public_mcp_url,
                    "transport": "streamable-http"
                }
            }
        }
        st.code(json.dumps(cursor_cfg, indent=2), language="json")

st.divider()
st.caption("Powered by Google Gemini Free Tier & Model Context Protocol (v2). Built with Streamlit.")
