import os
import json
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.generate import LLM_MODEL
from agent.prompts import CLASSIFICATION_PROMPT

def extract_json(text: str) -> dict:
    """
    Robustly extracts and parses a JSON object from response text.
    Handles potential markdown code blocks.
    """
    text = text.strip()
    
    # Strip markdown block if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Find index where the code starts
        start_idx = 1
        # If the first line is just ```json or ```, skip it
        if lines[0].strip().startswith("```"):
            start_idx = 1
        else:
            start_idx = 0
            
        # Find where it ends
        end_idx = len(lines)
        if lines[-1].strip() == "```":
            end_idx = -1
            
        text = "\n".join(lines[start_idx:end_idx]).strip()
        
    # In case there's still some surrounding text, find the first '{' and last '}'
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1:
        text = text[first_brace:last_brace + 1]
        
    return json.loads(text)

def classify(feedback_text: str, similar_examples: list[dict]) -> dict:
    """
    Calls Gemma to classify the customer feedback using similar examples as context.
    Respects the 15 RPM rate limit by sleeping 4 seconds before the call.
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")
        
    client = genai.Client(api_key=api_key)
    
    # Format the similar examples into a string
    examples_str = ""
    for idx, ex in enumerate(similar_examples, 1):
        examples_str += f"--- Example {idx} [Source: {ex.get('source', 'unknown')}] ---\n{ex.get('text', '')}\n\n"
        
    # Build prompt
    prompt = CLASSIFICATION_PROMPT.format(
        similar_examples=examples_str.strip(),
        feedback_text=feedback_text
    )
    
    # Standard fallback dict
    fallback = {
        "sentiment": "neutral",
        "category": "other",
        "urgency": "low",
        "recommended_team": "support",
        "reasoning": "Classification failed, defaulting"
    }
    
    # Sleep 4 seconds BEFORE the Gemma call to respect the 15 RPM rate limit
    time.sleep(4)
    
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0
            )
        )
        result = extract_json(response.text)
    except Exception as e:
        print(f"  [!] Classification failed/JSON parse error: {e}. Retrying once with strict instructions...")
        
        # Sleep 4 seconds BEFORE retry call
        time.sleep(4)
        
        strict_prompt = prompt + "\n\nREMINDER: You must return ONLY valid JSON. No markdown backticks, no introduction, and no explanation."
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=strict_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0
                )
            )
            result = extract_json(response.text)
        except Exception as retry_e:
            print(f"  [!] Strict retry also failed: {retry_e}. Using fallback.")
            return fallback
            
    # Guarantee keys exist
    for key, val in fallback.items():
        if key not in result:
            result[key] = val
            
    return result

if __name__ == "__main__":
    print("Testing classify()...")
    test_feedback = "I love this app, but it is slow"
    test_examples = [
        {"text": "App is good", "source": "synthetic"},
        {"text": "Very laggy interface", "source": "amazon"}
    ]
    try:
        res = classify(test_feedback, test_examples)
        print("Result:")
        print(res)
    except Exception as e:
        print(f"Error: {e}")
