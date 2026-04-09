"""
Hard Task: Multi-Concept Synthesis
====================================
Query: "What are the recent advances combining reinforcement learning
        with large language models for decision-making?"

The agent must search a complex multi-concept query, retrieve papers
across 4 sub-themes (RLHF/alignment, tool use, RL decision-making,
planning), synthesize connections between papers, and produce
explanations that show cross-paper relationships.
"""

from tasks.easy import TaskConfig


HARD_TASK = TaskConfig(
    name="multi_concept_synthesis",
    difficulty="hard",
    query=(
        "What are the recent advances combining reinforcement learning "
        "with large language models for decision-making?"
    ),
    ground_truth_paper_ids=[
        "paper_014",  # InstructGPT (RLHF)
        "paper_016",  # DPO
        "paper_017",  # PPO
        "paper_021",  # Toolformer
        "paper_022",  # ReAct
        "paper_028",  # Decision Transformer
        "paper_029",  # LLMs as Decision Makers
        "paper_030",  # RL + Foundation Models Survey
    ],
    required_sub_topics=[
        "rlhf_alignment",
        "llm_tool_use",
        "rl_decision_making",
        "llm_planning",
    ],
    reference_keywords={
        "paper_014": [
            "RLHF", "human feedback", "reward model", "PPO",
            "alignment", "instruction following", "InstructGPT",
        ],
        "paper_016": [
            "DPO", "preference optimization", "reward model",
            "policy", "alignment", "simpler",
        ],
        "paper_017": [
            "PPO", "policy gradient", "clipped objective",
            "trust region", "stability", "reinforcement learning",
        ],
        "paper_021": [
            "Toolformer", "tool use", "API calls", "self-supervised",
            "search", "calculator", "external tools",
        ],
        "paper_022": [
            "ReAct", "reasoning", "acting", "chain-of-thought",
            "grounded", "interleaving", "hallucination",
        ],
        "paper_028": [
            "Decision Transformer", "sequence modeling", "offline RL",
            "return-to-go", "causal transformer",
        ],
        "paper_029": [
            "LLM", "decision making", "interactive", "text-based",
            "planning", "grounding", "zero-shot",
        ],
        "paper_030": [
            "foundation models", "world models", "reward signals",
            "policy", "pre-training", "distribution shift",
        ],
    },
    explanation_keywords={
        "paper_014": [
            "feedback", "human", "reward", "helpful", "align",
        ],
        "paper_016": [
            "simpler", "preference", "directly", "skip", "stable",
        ],
        "paper_017": [
            "stable", "clipping", "update", "default", "algorithm",
        ],
        "paper_021": [
            "tools", "search", "calculator", "learn", "APIs",
        ],
        "paper_022": [
            "think", "act", "observe", "grounded", "real",
        ],
        "paper_028": [
            "language", "sequence", "predict", "action", "outcome",
        ],
        "paper_029": [
            "text", "environment", "reason", "decisions", "knowledge",
        ],
        "paper_030": [
            "foundation", "accelerate", "paradigms", "challenges",
        ],
    },
    cross_reference_required=True,
    grading_weights={
        "relevance": 0.20,
        "correctness": 0.20,
        "completeness": 0.25,
        "explanation_quality": 0.35,
    },
    min_summaries=5,
    min_explanations=3,
)
