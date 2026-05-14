"""
rag.py - The full RAG pipeline: question -> answer.
"""
import sys
import argparse
from src.retrieve import retrieve
from src.generate import generate_answer

def ask(question: str) -> dict:
    """
    Accepts a user question, retrieves relevant documents, 
    and generates an answer. Returns a dict with question, answer, and sources.
    """
    retrieved_docs = retrieve(question, top_k=5)
    
    if not retrieved_docs:
        return {
            "question": question,
            "answer": "I don't know (No relevant documents found).",
            "sources": []
        }
        
    answer = generate_answer(question, retrieved_docs)
    
    return {
        "question": question,
        "answer": answer,
        "sources": retrieved_docs
    }

def main():
    parser = argparse.ArgumentParser(description="Query the feedback-rag pipeline.")
    parser.add_argument("question", type=str, nargs="?", help="The question to ask about the reviews")
    args = parser.parse_args()

    if args.question:
        # Single query mode
        print(f"\nQuestion: {args.question}")
        print("Retrieving and generating answer...\n")
        result = ask(args.question)
        
        print(f"Answer:\n{result['answer']}\n")
        print("Sources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"[{i}] {source['metadata']['Summary']} (Score: {source['metadata']['Score']}/5)")
    else:
        # Interactive loop
        print("Welcome to feedback-rag! (Type 'exit' or 'quit' to end)")
        while True:
            try:
                question = input("\nQ: ")
                if question.strip().lower() in ['exit', 'quit']:
                    break
                if not question.strip():
                    continue
                    
                result = ask(question)
                print(f"\nA: {result['answer']}")
                print("\nSources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"[{i}] {source['metadata']['Summary']} (Score: {source['metadata']['Score']}/5)")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
