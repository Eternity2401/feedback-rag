"""
generate.py - Calls Gemini LLM with retrieved context + question to produce an answer.
"""
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

LLM_MODEL = "gemma-4-31b-it"

def generate_answer(question: str, retrieved_docs: list) -> str:
    """
    Calls Gemini gemini-2.0-flash with the provided question and retrieved context.
    Returns the generated answer.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)
    
    # Format the context from retrieved documents
    context_str = ""
    for i, doc in enumerate(retrieved_docs, 1):
        context_str += f"--- Review {i} ---\n{doc['document']}\n\n"
        
    system_instruction = (
        "You are an assistant answering questions based on Amazon product reviews. "
        "Answer ONLY from the provided reviews. Cite snippets from the reviews to support your answer. "
        "Say 'I don't know' if the reviews don't contain the answer."
    )
    
    prompt = f"Question: {question}\n\nContext:\n{context_str}"
    
    import time
    import random
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0
                )
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1 and any(kw in str(e).lower() for kw in ["429", "resource_exhausted", "quota"]):
                wait = min(2.0 ** attempt + random.uniform(0, 2), 60.0)
                print(f"  [!] Rate limit hit for generate (attempt {attempt + 1}/{max_retries}). Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise Exception(e)

if __name__ == "__main__":
    print("This module is typically called from rag.py")
