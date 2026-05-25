import os
import sys
import time
import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add parent dir to path if needed for imports
sys.path.append(str(Path(__file__).parent.parent))

from agent import retrieve
from agent import classify
from agent import route

def append_to_csv(filepath: Path, row_dict: dict):
    """
    Appends a single dictionary row to the target CSV file.
    Creates the file and writes the header if the file does not exist.
    """
    df = pd.DataFrame([row_dict])
    header = not filepath.exists()
    df.to_csv(filepath, mode='a', index=False, header=header)

def main():
    start_time = time.time()
    
    csv_path = Path("data/triage_feedbacks.csv")
    outputs_dir = Path("agent/outputs")
    results_path = outputs_dir / "triaged_results.csv"
    log_path = outputs_dir / "decisions.log"
    
    if not csv_path.exists():
        print(f"[ERROR] Source file not found: {csv_path}")
        sys.exit(1)
        
    # Create output directory
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Load completed IDs for resume safety
    completed_ids = set()
    if results_path.exists():
        try:
            completed_df = pd.read_csv(results_path)
            if "id" in completed_df.columns:
                completed_ids = set(completed_df["id"].astype(str).tolist())
                print(f"[OK] Resume-safe: skipping {len(completed_ids)} already triaged feedbacks.")
        except Exception as e:
            print(f"[WARNING] Could not read existing results: {e}. Starting fresh.")
            
    # Load source feedbacks
    try:
        source_df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to load {csv_path}: {e}")
        sys.exit(1)
        
    feedbacks_to_process = []
    for _, row in source_df.iterrows():
        fb_id = str(row["id"])
        if fb_id not in completed_ids:
            feedbacks_to_process.append(row)
            
    total_feedbacks = len(source_df)
    to_process_count = len(feedbacks_to_process)
    print(f"[OK] Found {total_feedbacks} total feedbacks. {to_process_count} left to process.")
    
    # Iterate with tqdm
    with tqdm(total=to_process_count, desc="Triaging Feedbacks") as pbar:
        for row in feedbacks_to_process:
            fb_id = int(row["id"])
            feedback_text = str(row["text"])
            
            try:
                # a. Retrieve 3 similar examples
                # Include print output in tqdm.write to avoid cluttering progress bar
                tqdm.write(f"\nProcessing ID {fb_id}...")
                similar_examples = retrieve.retrieve_similar(feedback_text, k=3)
                
                # b. Classify
                classification = classify.classify(feedback_text, similar_examples)
                
                # c. Route
                routing = route.route(classification)
                
                # Prepare CSV row
                similar_cols = ["", "", ""]
                for idx, ex in enumerate(similar_examples[:3]):
                    similar_cols[idx] = f"[{ex.get('source', 'unknown')}] (dist: {ex.get('distance', 0.0):.3f}) {ex.get('text', '')}"
                
                row_dict = {
                    "id": fb_id,
                    "text": feedback_text,
                    "sentiment": classification.get("sentiment", "neutral"),
                    "category": classification.get("category", "other"),
                    "urgency": classification.get("urgency", "low"),
                    "recommended_team": classification.get("recommended_team", "support"),
                    "destination": routing.get("destination", "support"),
                    "queue": routing.get("queue", "normal"),
                    "priority_score": routing.get("priority_score", 3),
                    "reasoning": classification.get("reasoning", ""),
                    "similar_1": similar_cols[0],
                    "similar_2": similar_cols[1],
                    "similar_3": similar_cols[2]
                }
                
                # d. Append to CSV
                append_to_csv(results_path, row_dict)
                
                # e. Append structured log entry
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"[{timestamp}] ID {fb_id} | TEXT: {feedback_text}\n")
                    lf.write(f"  CLASSIFICATION: {classification}\n")
                    lf.write(f"  ROUTING: {routing}\n")
                    lf.write("  RETRIEVED:\n")
                    for i, ex in enumerate(similar_examples, 1):
                        lf.write(f"    {i}. [{ex.get('source')}] (dist: {ex.get('distance', 0.0):.3f}) {ex.get('text')}\n")
                    lf.write(f"{'-'*80}\n")
                    
            except Exception as e:
                # Robust error handling: log and continue
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                tqdm.write(f"  [ERROR] Failed to process ID {fb_id}: {e}")
                try:
                    with open(log_path, "a", encoding="utf-8") as lf:
                        lf.write(f"[{timestamp}] ID {fb_id} | TEXT: {feedback_text}\n")
                        lf.write(f"  ERROR: {str(e)}\n")
                        lf.write(f"{'-'*80}\n")
                except Exception as log_e:
                    tqdm.write(f"  [CRITICAL] Failed to write error log: {log_e}")
                    
            pbar.update(1)
            
    elapsed_time = time.time() - start_time
    
    # 5. Print summary stats
    if results_path.exists():
        try:
            final_df = pd.read_csv(results_path)
            total_processed = len(final_df)
            
            print("\n" + "="*50)
            print("TRIAGE RUN SUMMARY STATS")
            print("="*50)
            print(f"Total processed feedbacks: {total_processed}")
            
            print("\nDistribution by Category:")
            category_counts = final_df["category"].value_counts()
            for cat, cnt in category_counts.items():
                print(f"  - {cat}: {cnt}")
                
            print("\nDistribution by Urgency:")
            urgency_counts = final_df["urgency"].value_counts()
            for urg, cnt in urgency_counts.items():
                print(f"  - {urg}: {cnt}")
                
            print("\nDistribution by Destination:")
            dest_counts = final_df["destination"].value_counts()
            for dest, cnt in dest_counts.items():
                print(f"  - {dest}: {cnt}")
                
            print(f"\nTotal runtime: {elapsed_time:.2f}s")
            print("="*50 + "\n")
        except Exception as e:
            print(f"[ERROR] Failed to display summary stats: {e}")
    else:
        print("[WARNING] No triaged results CSV was created.")

if __name__ == "__main__":
    main()
