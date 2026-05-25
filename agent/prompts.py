"""
prompts.py - Prompts and routing rules for the Triage Agent Module.
"""

CLASSIFICATION_PROMPT = """You are an expert customer feedback classification AI.
Your task is to analyze the following customer feedback along with 3 similar past entries (provided for context) and classify it.

Retrieve Past Context Reviews:
{similar_examples}

Customer Feedback to Classify:
"{feedback_text}"

Based on the feedback, return a JSON object containing exactly these keys:
- "sentiment": one of ["positive", "negative", "neutral", "mixed"]
- "category": one of ["bug", "feature_request", "praise", "complaint", "billing", "spam", "other"]
- "urgency": one of ["critical", "high", "medium", "low"]
- "recommended_team": one of ["engineering", "product", "support", "billing", "marketing", "trash"]
- "reasoning": a 1-2 sentence string explaining the classification

Crucial constraint: You MUST return ONLY valid JSON.
Do NOT wrap the output in markdown code fences (like ```json ... ```).
Do NOT include any introduction, explanations, or text before or after the JSON object.
"""

ROUTING_RULES = """
- bug + critical -> engineering, queue=P0
- bug + high -> engineering, queue=P1
- bug + medium/low -> engineering, queue=P2
- billing + any -> billing, queue=refunds
- feature_request + any -> product, queue=backlog
- praise + any -> marketing, queue=wins_board
- complaint + critical/high -> support, queue=priority
- complaint + medium/low -> support, queue=normal
- spam -> trash, queue=trash
"""
