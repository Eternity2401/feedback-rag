"""
server.py - Core Model Context Protocol (MCP) Server for feedback-rag.
Exposes tools, resources, and prompts over standard MCP v2 protocol.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from mcp.server.mcpserver import MCPServer
from mcp_server.rate_limiter import (
    rate_limiter,
    response_cache,
    validate_text_input,
    sanitize_k,
)

# Configure logging to stderr to keep stdout pure for stdio protocol
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp_server.server")

# Initialize MCP Server instance
mcp_server = MCPServer(
    name="feedback-rag",
    version="1.0.0",
    instructions="Standardized Model Context Protocol server exposing Amazon Reviews RAG and autonomous feedback triage capabilities.",
)


# ==============================================================================
# MCP TOOLS
# ==============================================================================

@mcp_server.tool()
def query_reviews_rag(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Executes a natural language RAG Q&A query against the Amazon Reviews ChromaDB vector store.
    Retrieves cited review context and generates a grounded response.

    Args:
        question: The user's question about product reviews or features.
        top_k: Number of relevant reviews to retrieve (1-10, default 5).
    """
    # 1. Validation & Rate Limiting
    val_err = validate_text_input(question, "question")
    if val_err:
        return {"success": False, "question": question, "answer": "", "sources": [], "error": val_err}

    allowed, limit_msg = rate_limiter.is_allowed("global")
    if not allowed:
        return {"success": False, "question": question, "answer": "", "sources": [], "error": limit_msg}

    k_clean = sanitize_k(top_k, default=5, max_k=10)
    cache_key = f"rag:{question.strip().lower()}:{k_clean}"

    cached = response_cache.get(cache_key)
    if cached:
        logger.info(f"Serving cached RAG result for query: {question[:50]}...")
        return cached

    try:
        # Import existing RAG pipeline without modification
        from src.rag import ask

        result = ask(question)
        sources_clean = []
        for s in result.get("sources", []):
            meta = s.get("metadata", {})
            sources_clean.append({
                "summary": meta.get("Summary", ""),
                "score": meta.get("Score", 0),
                "text": meta.get("Text", s.get("document", "")),
                "distance": float(s.get("distance", 0.0)) if s.get("distance") is not None else 0.0,
            })

        response = {
            "success": True,
            "question": question,
            "answer": result.get("answer", ""),
            "sources": sources_clean[:k_clean],
            "error": None,
        }
        response_cache.set(cache_key, response)
        return response

    except Exception as e:
        logger.error(f"Error executing query_reviews_rag: {e}", exc_info=True)
        err_msg = str(e).lower()
        if any(kw in err_msg for kw in ["429", "quota", "resource_exhausted"]):
            user_err = "Gemini free-tier quota temporarily reached. Please wait a moment and try again."
        else:
            user_err = "Unable to process RAG query at this time. Please try again shortly."
        return {"success": False, "question": question, "answer": "", "sources": [], "error": user_err}


@mcp_server.tool()
def semantic_search_reviews(query: str, k: int = 3) -> Dict[str, Any]:
    """
    Performs raw semantic vector similarity search against the ChromaDB collection.
    Returns ranked reviews with cosine distance metrics without invoking the LLM generator.

    Args:
        query: Search term or feedback phrase to find similar reviews for.
        k: Number of nearest review neighbors to return (1-10, default 3).
    """
    val_err = validate_text_input(query, "query")
    if val_err:
        return {"success": False, "query": query, "results": [], "error": val_err}

    allowed, limit_msg = rate_limiter.is_allowed("global")
    if not allowed:
        return {"success": False, "query": query, "results": [], "error": limit_msg}

    k_clean = sanitize_k(k, default=3, max_k=10)
    cache_key = f"search:{query.strip().lower()}:{k_clean}"

    cached = response_cache.get(cache_key)
    if cached:
        return cached

    try:
        from agent.retrieve import retrieve_similar

        results = retrieve_similar(query, k=k_clean)
        response = {
            "success": True,
            "query": query,
            "results": results,
            "error": None,
        }
        response_cache.set(cache_key, response)
        return response

    except Exception as e:
        logger.error(f"Error executing semantic_search_reviews: {e}", exc_info=True)
        return {"success": False, "query": query, "results": [], "error": "Semantic search temporarily unavailable."}


@mcp_server.tool()
def triage_customer_feedback(feedback_text: str) -> Dict[str, Any]:
    """
    Executes the autonomous feedback triage pipeline:
    1. Retrieves similar past context reviews from ChromaDB
    2. Classifies sentiment, category, urgency, and recommended team via Gemma
    3. Applies deterministic routing rules to assign destination, queue, and priority score.

    Args:
        feedback_text: Unstructured customer support message, review, or ticket text.
    """
    val_err = validate_text_input(feedback_text, "feedback_text")
    if val_err:
        return {"success": False, "feedback_text": feedback_text, "error": val_err}

    allowed, limit_msg = rate_limiter.is_allowed("global")
    if not allowed:
        return {"success": False, "feedback_text": feedback_text, "error": limit_msg}

    cache_key = f"triage:{feedback_text.strip().lower()}"
    cached = response_cache.get(cache_key)
    if cached:
        return cached

    try:
        from agent.retrieve import retrieve_similar
        from agent.classify import classify
        from agent.route import route

        # Step 1: Context retrieval
        similar_examples = retrieve_similar(feedback_text, k=3)

        # Step 2: LLM Classification
        classification = classify(feedback_text, similar_examples)

        # Step 3: Deterministic Routing
        routing = route(classification)

        response = {
            "success": True,
            "feedback_text": feedback_text,
            "sentiment": classification.get("sentiment", "neutral"),
            "category": classification.get("category", "other"),
            "urgency": classification.get("urgency", "low"),
            "destination": routing.get("destination", "support"),
            "queue": routing.get("queue", "normal"),
            "priority_score": routing.get("priority_score", 3),
            "reasoning": classification.get("reasoning", ""),
            "similar_reviews": similar_examples,
            "error": None,
        }
        response_cache.set(cache_key, response)
        return response

    except Exception as e:
        logger.error(f"Error in triage_customer_feedback: {e}", exc_info=True)
        return {
            "success": False,
            "feedback_text": feedback_text,
            "error": "Triage failed due to API rate limit or service unavailability.",
        }


@mcp_server.tool()
def get_triage_batch_metrics() -> Dict[str, Any]:
    """
    Read-only analytics tool that summarizes the latest batch triage run from agent/outputs/triaged_results.csv.
    Produces category, urgency, destination, and queue distributions without consuming LLM quota.
    """
    csv_path = Path("agent/outputs/triaged_results.csv")
    if not csv_path.exists():
        return {
            "success": False,
            "error": "Triage metrics file is not available.",
        }

    try:
        df = pd.read_csv(csv_path)
        total = len(df)
        cat_dist = df["category"].value_counts().to_dict() if "category" in df.columns else {}
        urg_dist = df["urgency"].value_counts().to_dict() if "urgency" in df.columns else {}
        dest_dist = df["destination"].value_counts().to_dict() if "destination" in df.columns else {}
        queue_dist = df["queue"].value_counts().to_dict() if "queue" in df.columns else {}
        avg_priority = float(df["priority_score"].mean()) if "priority_score" in df.columns else 0.0

        return {
            "success": True,
            "total_feedback": total,
            "category_distribution": cat_dist,
            "urgency_distribution": urg_dist,
            "destination_distribution": dest_dist,
            "queue_distribution": queue_dist,
            "average_priority_score": round(avg_priority, 2),
            "error": None,
        }
    except Exception as e:
        logger.error(f"Error reading triage metrics: {e}", exc_info=True)
        return {"success": False, "error": f"Error parsing metrics: {str(e)}"}


# ==============================================================================
# MCP RESOURCES
# ==============================================================================

@mcp_server.resource("reviews://dataset-summary")
def get_dataset_summary() -> str:
    """Read-only resource describing the ChromaDB review dataset configuration."""
    summary_data = {
        "dataset_name": "Amazon Fine Food Reviews (Sampled)",
        "total_documents_ingested": 850,
        "chroma_collection": "reviews",
        "embedding_model": "gemini-embedding-001 / gemini-embedding-2",
        "embedding_dimensions": 3072,
        "distance_metric": "cosine",
        "metadata_fields": ["Score", "Summary", "Text", "source"],
        "purpose": "Retrieval-Augmented Q&A and semantic neighbor context grounding.",
    }
    return json.dumps(summary_data, indent=2)


@mcp_server.resource("triage://queues")
def get_triage_queues() -> str:
    """Read-only resource detailing the deterministic team queue structure and routing policy."""
    queue_data = {
        "routing_policy": "Deterministic Category x Urgency Rule Engine",
        "teams": {
            "engineering": {
                "queues": ["P0", "P1", "P2"],
                "triggers": "Category: bug (urgency: critical -> P0, high -> P1, medium/low -> P2)",
            },
            "billing": {
                "queues": ["refunds"],
                "triggers": "Category: billing (any urgency)",
            },
            "product": {
                "queues": ["backlog"],
                "triggers": "Category: feature_request (any urgency)",
            },
            "marketing": {
                "queues": ["wins_board"],
                "triggers": "Category: praise (any urgency)",
            },
            "support": {
                "queues": ["priority", "normal"],
                "triggers": "Category: complaint (urgency: critical/high -> priority, medium/low -> normal)",
            },
            "trash": {
                "queues": ["trash"],
                "triggers": "Category: spam (priority score: 0)",
            },
        },
    }
    return json.dumps(queue_data, indent=2)


# ==============================================================================
# MCP PROMPTS
# ==============================================================================

@mcp_server.prompt("feedback_triage_prompt")
def feedback_triage_prompt(feedback_text: str = "") -> str:
    """Standardized client prompt template for feedback triage."""
    return (
        f"You are an autonomous customer feedback triage specialist.\n\n"
        f"Analyze the following feedback and classify it into: "
        f"sentiment (positive/negative/neutral/mixed), "
        f"category (bug/feature_request/praise/complaint/billing/spam/other), "
        f"urgency (critical/high/medium/low), and recommended_team.\n\n"
        f"Customer Feedback:\n\"{feedback_text}\"\n\n"
        f"Provide classification and concise reasoning."
    )


# ==============================================================================
# CLI STDIO ENTRYPOINT
# ==============================================================================

def main():
    """Runs the MCP server over standard input/output (stdio)."""
    logger.info("Starting feedback-rag MCP Server in stdio transport mode...")
    mcp_server.run("stdio")


if __name__ == "__main__":
    main()
