"""
run_eval.py - Runs the eval set against the RAG pipeline, prints accuracy + cost.
"""
import os
import sys
import json
import time
from pathlib import Path

# Add src to the path so we can import from it
sys.path.append(str(Path(__file__).parent.parent))
from src.rag import ask

EVAL_SET_PATH = Path("evals/eval_set.json")
RESULTS_PATH = Path("evals/results.json")

def evaluate_answer(answer: str, expected_themes: list, must_mention: list) -> dict:
    """
    Scores the answer based on keyword and theme coverage.
    """
    answer_lower = answer.lower()
    
    # Check if the model punted
    if "i don't know" in answer_lower or "no relevant documents" in answer_lower:
        return {
            "score": 0.0,
            "must_mention_coverage": 0.0,
            "themes_coverage": 0.0,
            "punted": True
        }
        
    # Check must_mention coverage (any of the keywords)
    mentioned_keywords = [kw for kw in must_mention if kw.lower() in answer_lower]
    must_mention_coverage = len(mentioned_keywords) / len(must_mention) if must_mention else 1.0
    
    # Check themes coverage (soft match, any of the themes)
    mentioned_themes = [theme for theme in expected_themes if theme.lower() in answer_lower]
    themes_coverage = len(mentioned_themes) / len(expected_themes) if expected_themes else 1.0
    
    # Simple score: average of the two coverages
    score = (must_mention_coverage + themes_coverage) / 2.0
    
    return {
        "score": score,
        "must_mention_coverage": must_mention_coverage,
        "themes_coverage": themes_coverage,
        "punted": False,
        "mentioned_keywords": mentioned_keywords,
        "mentioned_themes": mentioned_themes
    }

def main():
    if not EVAL_SET_PATH.exists():
        print(f"Error: Eval set not found at {EVAL_SET_PATH}")
        sys.exit(1)
        
    with open(EVAL_SET_PATH, "r") as f:
        eval_data = json.load(f)
        
    print(f"Starting evaluation of {len(eval_data)} questions...\n")
    
    results = []
    total_score = 0.0
    punted_count = 0
    total_time = 0.0
    
    for i, item in enumerate(eval_data, 1):
        question = item["question"]
        print(f"[{i}/{len(eval_data)}] Q: {question}")
        
        answer = None
        rag_result = None
        
        # Retry loop for rate limits
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                rag_result = ask(question)
                answer = rag_result["answer"]
                elapsed = time.time() - start_time
                total_time += elapsed
                break # Success
            except Exception as e:
                err_str = str(e).lower()
                is_retryable = any(kw in err_str for kw in ["429", "resource_exhausted", "quota", "500", "internal"])
                if is_retryable and attempt < max_retries:
                    print(f"  [!] Retryable error hit (attempt {attempt+1}/{max_retries+1}). Sleeping 60s...")
                    time.sleep(60)
                else:
                    print(f"  [ERROR] Final failure after {attempt+1} attempts: {e}")
                    answer = "Error: Rate limit exhausted"
                    rag_result = {"sources": []}
                    elapsed = 0.0
                    break
        
        eval_metrics = evaluate_answer(answer, item["expected_themes"], item["must_mention"])
        
        score = eval_metrics["score"]
        total_score += score
        if eval_metrics["punted"]:
            punted_count += 1
            
        print(f"  A: {answer[:100]}...")
        print(f"  Score: {score:.2f} | Punted: {eval_metrics['punted']} | Time: {elapsed:.2f}s\n")
        
        result_item = {
            "id": item["id"],
            "question": question,
            "answer": answer,
            "sources_count": len(rag_result.get("sources", [])),
            "expected_themes": item["expected_themes"],
            "must_mention": item["must_mention"],
            "metrics": eval_metrics,
            "time_seconds": elapsed
        }
        results.append(result_item)
        
        # Sleep to avoid rate limits
        if i < len(eval_data):
            time.sleep(20)
            
    # Calculate summary
    avg_score = total_score / len(eval_data)
    
    summary = {
        "total_questions": len(eval_data),
        "average_score": avg_score,
        "punted_count": punted_count,
        "answer_rate": (len(eval_data) - punted_count) / len(eval_data),
        "total_time_seconds": total_time,
        "cost": "Free (Gemini Free Tier)"
    }
    
    print("="*40)
    print("EVALUATION SUMMARY")
    print("="*40)
    print(f"Total Questions: {summary['total_questions']}")
    print(f"Average Score:   {summary['average_score']:.2f} (0.0 to 1.0)")
    print(f"Answer Rate:     {summary['answer_rate']*100:.1f}% ({len(eval_data) - punted_count} answered, {punted_count} punted)")
    print(f"Total Time:      {summary['total_time_seconds']:.2f}s")
    print(f"Cost:            {summary['cost']}")
    print("="*40)
    
    # Save results
    output_data = {
        "summary": summary,
        "results": results
    }
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\nDetailed results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
