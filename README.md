# Feedback RAG: Amazon Product Review Analyzer

This project implements a complete Retrieval-Augmented Generation (RAG) pipeline to analyze Amazon product reviews. It uses Google's latest embedding and generation models (Free Tier) to provide factual, cited answers to complex consumer questions.

## Features

- **Ingestion**: Efficiently chunks and embeds CSV data into a local ChromaDB vector store.
- **Retrieval**: Performs semantic search using `gemini-embedding-001` to find the top-5 most relevant reviews.
- **Generation**: Produces multi-point, cited answers using the `Gemma-3-27B-IT` model.
- **Evaluation**: Includes a custom evaluation suite to measure accuracy, must-mention coverage, and theme alignment.
- **Resilience**: Robust retry logic with exponential backoff to navigate strict free-tier rate limits.

---

## 📝 Sample Queries

Below are actual outputs from the evaluation suite showing the system's ability to synthesize multiple reviews into a coherent answer.

### Sample 1: Packaging Complaints
**Question**: *Are the chips usually crushed upon arrival?*

**Answer**:
> Experiences vary among reviewers:
> 
> * Some report that the chips arrive damaged, with one reviewer stating "most of the chips were broken" (Review 1) and another noting that "some bags end up as crumbs" (Review 3).
> * Others report they arrive in good condition, stating "All the chips are as they should be" (Review 2), "Chips are never crushed or broken" (Review 4), and that they "came packaged perfectly" (Review 5).

### Sample 2: Dietary Information
**Question**: *Is this product gluten-free?*

**Answer**:
> Yes, the product is gluten-free. Reviewers state that they buy the product "because they are gluten free" (Review 1), describe it as "Gluten Free Oatmeal" (Review 2), call them "my favorite gluten-free dairy-free flavored chips" (Review 3), note it is "Great for Gluten-free lifestyle!!" (Review 4), and mention "It's also nice to have something gluten-free for friends" (Review 5).

---

## Tech Stack

- **Embeddings:** Google Gemini `gemini-embedding-2` (3072 dims, free tier)
- **Vector Store:** ChromaDB (local persistent)
- **LLM:** Google Gemma `Gemma-27B-IT` (Open weights, free tier via SDK)
- **Data Handling:** pandas

## Evaluation Results

**Current Status**: Evaluation complete for the initial 850-document set using **Gemma-27B-IT** and robust retry logic.

- **Total Documents Ingested:** 850
- **Average Accuracy (Score):** 39% 
- **Category Breakdown (Avg Score):**
  - **Comparison:** 47%
  - **Complaint:** 49%
  - **Sentiment:** 36%
  - **Information:** 29%
- **"I don't know" Punts:** 3
- **Rate-limit Failures:** 0/20 (Multi-layer retry logic successfully bypassed all 429 errors)
- **Latency:** ~20s / query (including backoff delays)
- **Cost per query:** Free (Gemini Free Tier)

*Note: The system now achieves 100% completion on benchmarks by automatically retrying on rate limits and internal server errors. The lower "Information" score reflects the system correctly punting (answering "I don't know") when the 850-document subset lacks specific factual details.*

## 🚀 What I'd Improve With More Time

- **Full Scale Ingestion**: Ingest the full 568K document dataset (currently limited to an 850-doc sample due to free-tier daily quotas).
- **Boost Information Accuracy**: Implement a multi-step "Verification" agent to double-check factual claims in the Information category.
- **Reranking Layer**: Add a Cross-Encoder (e.g., Cohere or BGE-Reranker) to refine the top-5 results before passing them to the LLM.
- **Hybrid Search**: Combine BM25 keyword matching with vector search to better handle specific terms like brand names and product IDs.
- **Query Expansion**: Use multi-query generation to expand the user's initial question into 3 variations for broader retrieval.
- **Streaming Responses**: Implement Server-Sent Events (SSE) to show the LLM answer as it generates, improving perceived latency.
- **Prompt A/B Testing**: Systematically test different system instructions (e.g., Chain-of-Thought vs. Direct Answer) to maximize accuracy.
- **Fine-tuned Embeddings**: Fine-tune the embedding model on consumer electronics or food reviews to improve semantic capture of industry-specific jargon.

## 💰 Cost Analysis

The current project runs entirely on the **Google Free Tier**. Below is an analysis of estimated costs if moved to the Pay-as-you-go (Paid) tier.

### Assumptions:
- **Avg. Context Size**: 2,500 input tokens per query (5 retrieved reviews).
- **Avg. Answer Size**: 150 output tokens.
- **Embedding Density**: 500 tokens per document.

### Pricing:
| Model | Input (per 1M tokens) | Output (per 1M tokens) |
| --- | --- | --- |
| Gemini Embedding | $0.025 | N/A |
| Gemini 2.0 Flash | $0.10 | $0.40 |
| Gemma-3-27B (Estimated) | $0.15 | $0.60 |

### Scaling Projections:
| Volume | Math | Estimated Total Cost |
| --- | --- | --- |
| **Ingestion (10k docs)** | (10k * 500 tokens) * $0.025/1M | **$0.12** |
| **100 Queries** | 100 * (($0.10 * 2.5k) + ($0.40 * 0.15k)) / 1k | **$0.03** |
| **1,000 Queries** | 1,000 * ~$0.0003 | **$0.31** |
| **100,000 Queries** | 100,000 * ~$0.0003 | **$31.00** |

## Known Limitations

- **Gemini Free-Tier Rate Limits:** The free tier for `gemini-flash-latest` enforces a 15 RPM (Requests Per Minute) limit. Batch evaluations often hit this limit, requiring aggressive retry logic and delays.
- **Embedding Daily Quota:** There is a 1,000 RPD (Requests Per Day) limit for embedding models, which restricts single-day ingestion to ~900 documents (including search queries).
- **Model Aggregation:** `gemini-embedding-2` aggregates multiple inputs into a single vector if passed as a list. The ingestion script handles this by processing documents sequentially.

## License
MIT
