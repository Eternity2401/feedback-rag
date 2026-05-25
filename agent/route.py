def route(classification: dict) -> dict:
    """
    Applies the routing rules based on the feedback classification.
    Returns a dict with destination, queue, and priority_score.
    """
    category = str(classification.get("category", "other")).strip().lower()
    urgency = str(classification.get("urgency", "low")).strip().lower()
    
    # Priority mapping
    priority_map = {
        "critical": 10,
        "high": 7,
        "medium": 5,
        "low": 3,
        "spam": 0
    }
    
    # Base priority score
    if category == "spam":
        priority_score = 0
    else:
        priority_score = priority_map.get(urgency, 3)
        
    # Hardcoded routing rules
    destination = "support"
    queue = "normal"
    
    if category == "bug":
        destination = "engineering"
        if urgency == "critical":
            queue = "P0"
        elif urgency == "high":
            queue = "P1"
        else:  # medium or low
            queue = "P2"
            
    elif category == "billing":
        destination = "billing"
        queue = "refunds"
        
    elif category == "feature_request":
        destination = "product"
        queue = "backlog"
        
    elif category == "praise":
        destination = "marketing"
        queue = "wins_board"
        
    elif category == "complaint":
        destination = "support"
        if urgency in ("critical", "high"):
            queue = "priority"
        else:
            queue = "normal"
            
    elif category == "spam":
        destination = "trash"
        queue = "trash"
        priority_score = 0
        
    else:
        # Fallback for "other" or anything unspecified
        rec_team = str(classification.get("recommended_team", "support")).strip().lower()
        valid_teams = ["engineering", "product", "support", "billing", "marketing", "trash"]
        destination = rec_team if rec_team in valid_teams else "support"
        
        # Determine fallback queue
        if destination == "engineering":
            queue = "P2"
        elif destination == "billing":
            queue = "refunds"
        elif destination == "product":
            queue = "backlog"
        elif destination == "marketing":
            queue = "wins_board"
        elif destination == "trash":
            queue = "trash"
            priority_score = 0
        else:
            if urgency in ("critical", "high"):
                queue = "priority"
            else:
                queue = "normal"
                
    return {
        "destination": destination,
        "queue": queue,
        "priority_score": priority_score
    }

if __name__ == "__main__":
    print("Testing route()...")
    test_class = {"category": "bug", "urgency": "critical"}
    print(route(test_class)) # engineering, P0, 10
