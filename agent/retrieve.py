import os
import time
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.retrieve import EMBED_MODEL, CHROMA_DIR, COLLECTION

def retrieve_similar(feedback_text: str, k: int = 3) -> list[dict]:
    """
    Embeds the feedback_text and queries ChromaDB for the top-k most similar entries.
    Includes time.sleep(13) BEFORE the embedding call to respect the 5 RPM limit.
    """
    # Respect the 5 RPM rate limit of gemini-embedding-001 / gemini-embedding-2 in free tier
    time.sleep(13)
    
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)
    
    # Embed the feedback text
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[feedback_text],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        )
    )
    query_embedding = result.embeddings[0].values
    
    # Connect to persistent ChromaDB collection
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(name=COLLECTION)
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    # Format and return results
    retrieved_docs = []
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            meta = results['metadatas'][0][i] or {}
            # Check for source, default to "amazon" for the reviews
            source = meta.get("source", "amazon")
            # Extract text from metadata Text field if present, otherwise fallback to full document
            text = meta.get("Text", results['documents'][0][i])
            
            distance = float(results['distances'][0][i]) if (results.get('distances') and results['distances'][0]) else 0.0
            
            retrieved_docs.append({
                "text": text,
                "source": source,
                "distance": distance
            })
            
    return retrieved_docs

if __name__ == "__main__":
    import pprint
    print("Testing retrieve_similar('This app crashes constant on Android')...")
    try:
        res = retrieve_similar("This app crashes constant on Android", k=3)
        pprint.pprint(res)
    except Exception as e:
        print(f"Error: {e}")
