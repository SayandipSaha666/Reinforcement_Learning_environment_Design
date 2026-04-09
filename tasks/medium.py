"""
Medium Task: Ambiguous Query Filtering
========================================
Query: "How do neural networks handle uncertainty?"

The agent must search an ambiguous query that returns results across
multiple sub-topics (Bayesian NNs, dropout-as-inference, ensembles, calibration).
It must filter down to the truly relevant papers, summarize papers from at
least 2 different approaches, and explain the trade-offs.
"""

from tasks.easy import TaskConfig


MEDIUM_TASK = TaskConfig(
    name="ambiguous_query_filtering",
    difficulty="medium",
    query="How do neural networks handle uncertainty?",
    ground_truth_paper_ids=[
        "paper_008",  # Bayesian by Backprop
        "paper_009",  # MC Dropout
        "paper_010",  # Deep Ensembles
        "paper_011",  # Evidential DL
        "paper_012",  # Calibration
    ],
    required_sub_topics=[
        "bayesian_uncertainty",
        "dropout_uncertainty",
        "ensemble_uncertainty",
    ],
    reference_keywords={
        "paper_008": [
            "Bayesian", "weight distributions", "variational inference",
            "posterior", "uncertainty quantification", "backpropagation",
        ],
        "paper_009": [
            "dropout", "Bayesian approximation", "MC dropout",
            "Gaussian process", "forward passes", "epistemic",
        ],
        "paper_010": [
            "ensembles", "random initialization", "calibration",
            "out-of-distribution", "predictive uncertainty",
        ],
        "paper_011": [
            "evidential", "Dirichlet", "aleatoric", "epistemic",
            "single forward pass", "uncertainty decomposition",
        ],
        "paper_012": [
            "calibration", "overconfidence", "temperature scaling",
            "reliability", "confidence",
        ],
    },
    explanation_keywords={
        "paper_008": [
            "range", "values", "sure", "prediction", "uncertainty",
        ],
        "paper_009": [
            "dropout", "training", "predictions", "free", "multiple",
        ],
        "paper_010": [
            "copies", "model", "disagree", "average", "unusual",
        ],
        "paper_011": [
            "evidence", "noisy", "type", "uncertainty", "fast",
        ],
        "paper_012": [
            "confident", "right", "fix", "trustworthy",
        ],
    },
    cross_reference_required=False,
    grading_weights={
        "relevance": 0.25,
        "correctness": 0.25,
        "completeness": 0.25,
        "explanation_quality": 0.25,
    },
    min_summaries=3,
    min_explanations=2,
)
