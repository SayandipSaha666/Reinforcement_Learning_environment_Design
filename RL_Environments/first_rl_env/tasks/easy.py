"""
Easy Task: Single Topic Retrieval
==================================
Query: "What is the transformer architecture in NLP?"

The agent must find papers about transformers/attention in NLP,
filter to the most relevant ones, summarize 1-2 papers, and explain.

Ground truth has 3 clearly relevant papers with high relevance scores.
Distractor papers (CV, unrelated NLP) have low relevance.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TaskConfig:
    """Configuration for a single task."""
    name: str
    difficulty: str
    query: str
    ground_truth_paper_ids: List[str]
    required_sub_topics: List[str]
    reference_keywords: Dict[str, List[str]]  # paper_id -> expected keywords in summary
    explanation_keywords: Dict[str, List[str]]  # paper_id -> expected keywords in explanation
    cross_reference_required: bool  # whether explanation must connect papers
    grading_weights: Dict[str, float]
    min_summaries: int  # minimum papers to summarize for full completeness
    min_explanations: int  # minimum papers to explain for full completeness


EASY_TASK = TaskConfig(
    name="single_topic_retrieval",
    difficulty="easy",
    query="What is the transformer architecture in NLP?",
    ground_truth_paper_ids=["paper_001", "paper_003", "paper_007"],
    required_sub_topics=["transformer_architecture"],
    reference_keywords={
        "paper_001": [
            "self-attention", "multi-head", "encoder-decoder", "parallel",
            "positional encoding", "sequence", "translation",
        ],
        "paper_003": [
            "GPT", "autoregressive", "few-shot", "in-context learning",
            "scaling", "language model", "parameters",
        ],
        "paper_007": [
            "positional encoding", "sinusoidal", "RoPE", "relative position",
            "length generalization",
        ],
    },
    explanation_keywords={
        "paper_001": [
            "attention", "parallel processing", "sequence modeling",
            "words", "relationships",
        ],
        "paper_003": [
            "large", "examples", "prompt", "tasks", "scale",
        ],
        "paper_007": [
            "order", "position", "longer", "encoding",
        ],
    },
    cross_reference_required=False,
    grading_weights={
        "relevance": 0.30,
        "correctness": 0.30,
        "completeness": 0.20,
        "explanation_quality": 0.20,
    },
    min_summaries=1,
    min_explanations=1,
)
