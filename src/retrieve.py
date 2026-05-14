"""
retrieve.py - Given a question, returns top-k relevant reviews from ChromaDB.
"""
import os
from pathlib import Path
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

CHROMA_DIR = Path("chroma_db")
COLLECTION = "reviews"
EMBED_MODEL = "gemini-embedding-2"

def retrieve(query: str, top_k: int = 5) -> list:
    """
    Connects to persistent ChromaDB collection, embeds the user question,
    queries ChromaDB for top-k most similar documents, and returns results.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)
    
    # Embed the query
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        )
    )
    query_embedding = result.embeddings[0].values
    
    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(name=COLLECTION)
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Format the results
    retrieved_docs = []
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            doc = {
                "id": results['ids'][0][i],
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results and results['distances'] else None
            }
            retrieved_docs.append(doc)
            
    return retrieved_docs

if __name__ == "__main__":
    import pprint
    print("Testing retrieve('complaints about packaging')...")
    try:
        results = retrieve("complaints about packaging")
        pprint.pprint(results)
    except Exception as e:
        print(f"Error during retrieval: {e}")
