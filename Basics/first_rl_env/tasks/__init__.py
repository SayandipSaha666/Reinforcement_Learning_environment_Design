"""
Task definitions for the Research Paper Assistant environment.

Each task specifies:
- query: the research question
- ground_truth_paper_ids: papers that are truly relevant
- required_sub_topics: topic areas the agent should cover
- reference_keywords: key concepts that should appear in summaries
- grading_weights: per-metric weights for final score computation
"""

from .easy import EASY_TASK
from .medium import MEDIUM_TASK
from .hard import HARD_TASK

TASK_REGISTRY = {
    "single_topic_retrieval": EASY_TASK,
    "ambiguous_query_filtering": MEDIUM_TASK,
    "multi_concept_synthesis": HARD_TASK,
}


def get_task(task_name: str):
    """Get a task configuration by name."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_name]


__all__ = ["TASK_REGISTRY", "get_task", "EASY_TASK", "MEDIUM_TASK", "HARD_TASK"]
