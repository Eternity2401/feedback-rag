"""
ingest.py - Loads reviews.csv, embeds with Gemini models, stores in ChromaDB.

Usage:
    python -m src.ingest
"""

import os
import sys
import time
import random
from pathlib import Path

import pandas as pd
import chromadb
from google import genai
from google.genai import types
from tqdm import tqdm
from dotenv import load_dotenv

# -- Constants -----------------------------------------------------------------

CSV_PATH      = Path("data/reviews.csv")
CHROMA_DIR    = Path("chroma_db")
COLLECTION    = "reviews"
EMBED_MODEL   = "gemini-embedding-2" # Using sequential processing to avoid aggregation
MAX_ROWS      = 2000
BATCH_SIZE    = 50      
BATCH_DELAY   = 30.0    # 30s delay between batches (very safe for free tier)
MAX_RETRIES   = 10      
BACKOFF_BASE  = 5.0     
BACKOFF_CAP   = 180.0   


# -- Helpers -------------------------------------------------------------------

def embed_batch_with_backoff(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of texts using Gemini gemini-embedding-001.
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )
            return [emb.values for emb in result.embeddings]
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = any(kw in err_str for kw in ("429", "resource_exhausted", "quota", "rate"))
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                wait = min(BACKOFF_BASE ** attempt + random.uniform(0, 5), BACKOFF_CAP)
                tqdm.write(f"  [!] Rate limit. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise
    return []


def build_document(row: pd.Series) -> str:
    """Formats one CSV row into the document string used for embedding + retrieval."""
    return (
        f"Rating: {row['Score']}/5 | "
        f"Summary: {row['Summary']} | "
        f"Review: {row['Text']}"
    )


# -- Main ----------------------------------------------------------------------

def main() -> None:

    # 1. Load .env and validate API key
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("[ERROR] GOOGLE_API_KEY not found.")

    client = genai.Client(api_key=api_key)
    print(f"[OK] Gemini client ready | model: {EMBED_MODEL}")

    # 2. Load CSV
    if not CSV_PATH.exists():
        sys.exit(f"[ERROR] CSV not found.")

    print(f"[..] Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, nrows=MAX_ROWS)
    df = df[["Score", "Summary", "Text"]].dropna().reset_index(drop=True)
    total = len(df)
    print(f"[OK] Rows to ingest: {total}")

    # 3. Set up ChromaDB
    CHROMA_DIR.mkdir(exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    
    # Fetch existing IDs
    existing_data = collection.get(include=[])
    existing_ids = set(existing_data["ids"])
    print(f"[OK] Found {len(existing_ids)} existing documents.")

    # 4. Pre-build all documents
    documents = [build_document(row) for _, row in df.iterrows()]

    # 5. Embed in batches
    ingested = 0
    with tqdm(total=total, unit="doc", desc="Embedding") as pbar:
        # Initial update for existing
        pbar.update(len(existing_ids))
        
        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch_indices = list(range(batch_start, batch_end))
            indices_to_process = [i for i in batch_indices if str(i) not in existing_ids]
            
            if not indices_to_process:
                # Already updated pbar for existing initially
                continue
            
            batch_docs = [documents[i] for i in indices_to_process]
            batch_df   = df.iloc[indices_to_process]

            embeddings = embed_batch_with_backoff(client, batch_docs)

            ids = [str(i) for i in indices_to_process]
            metadatas = [
                {
                    "Score": int(row["Score"]),
                    "Summary": str(row["Summary"]),
                    "Text": str(row["Text"]),
                }
                for _, row in batch_df.iterrows()
            ]

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=batch_docs,
                metadatas=metadatas,
            )

            ingested += len(batch_docs)
            pbar.update(len(batch_docs))

            if batch_start + BATCH_SIZE < total:
                time.sleep(BATCH_DELAY)

    print(f"\n[DONE] Ingested {ingested} new documents.")


def ingest_synthetic_feedbacks(csv_path: str = "data/triage_feedbacks.csv") -> None:
    """
    Ingest synthetic feedbacks into the existing ChromaDB with source='synthetic' metadata.
    Resume-safe: skips IDs already in the collection.
    Respects the 5 RPM embedding limit via time.sleep(13) between calls.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY not found.")
        return

    client = genai.Client(api_key=api_key)

    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"[ERROR] CSV not found at {csv_path}")
        return

    print(f"[..] Loading synthetic feedbacks from {csv_path} ...")
    df = pd.read_csv(csv_file)
    
    # Set up ChromaDB (using same setup as main)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Fetch existing IDs
    existing_data = collection.get(include=[])
    existing_ids = set(existing_data["ids"])

    newly_ingested = 0
    total_feedbacks = len(df)
    
    print(f"[..] Processing {total_feedbacks} synthetic feedbacks...")
    
    for _, row in df.iterrows():
        fb_id = row["id"]
        doc_id = f"synthetic_{fb_id}"
        text = str(row["text"])
        
        if doc_id in existing_ids:
            continue
            
        print(f"  [+] Embedding and adding ID {doc_id} ...")
        
        # Respect the 5 RPM embedding limit via sleep(13) before the embedding call
        time.sleep(13)
        
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=[text],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                ),
            )
            embedding = result.embeddings[0].values
            
            # Add to the collection
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{"source": "synthetic"}]
            )
            newly_ingested += 1
            
        except Exception as e:
            print(f"  [ERROR] Failed to ingest ID {doc_id}: {e}")
            
    print(f"[DONE] Ingested {newly_ingested} new synthetic feedbacks.")


if __name__ == "__main__":
    main()

