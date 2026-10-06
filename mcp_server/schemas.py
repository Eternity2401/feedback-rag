"""
schemas.py - Structured request and response models for MCP tools.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RAGQueryResponse(BaseModel):
    success: bool = True
    question: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


class SemanticSearchResultItem(BaseModel):
    text: str
    source: str = "amazon"
    distance: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SemanticSearchResponse(BaseModel):
    success: bool = True
    query: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


class TriageResponse(BaseModel):
    success: bool = True
    feedback_text: str
    sentiment: str = "neutral"
    category: str = "other"
    urgency: str = "low"
    destination: str = "support"
    queue: str = "normal"
    priority_score: int = 3
    reasoning: str = ""
    similar_reviews: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


class BatchMetricsResponse(BaseModel):
    success: bool = True
    total_feedback: int = 0
    category_distribution: Dict[str, int] = Field(default_factory=dict)
    urgency_distribution: Dict[str, int] = Field(default_factory=dict)
    destination_distribution: Dict[str, int] = Field(default_factory=dict)
    queue_distribution: Dict[str, int] = Field(default_factory=dict)
    average_priority_score: float = 0.0
    error: Optional[str] = None
