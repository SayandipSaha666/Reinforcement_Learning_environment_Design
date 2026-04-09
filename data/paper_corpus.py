"""
Simulated Research Paper Corpus
================================
50 research papers spanning ML/AI topics organized by sub-topic clusters.
Each paper has: id, title, abstract, keywords, sub_topic, reference_summary,
reference_explanation, and relevance tags for each task.

This corpus is deterministic and embedded in-memory — no external API calls.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Paper:
    """A simulated research paper with ground-truth metadata."""
    id: str
    title: str
    abstract: str
    keywords: List[str]
    sub_topic: str
    reference_summary: str
    reference_explanation: str
    relevance_tags: Dict[str, float] = field(default_factory=dict)
    # relevance_tags maps task_name -> relevance_score (0.0 to 1.0)


# ---------------------------------------------------------------------------
# CORPUS: 50 papers across 8 sub-topics
# ---------------------------------------------------------------------------

PAPER_CORPUS: List[Paper] = [
    # =====================================================================
    # SUB-TOPIC 1: Transformer Architecture & Attention (papers 001-007)
    # =====================================================================
    Paper(
        id="paper_001",
        title="Attention Is All You Need: The Transformer Architecture",
        abstract="We propose a novel sequence-to-sequence architecture based entirely on attention mechanisms, dispensing with recurrence and convolutions. The Transformer uses multi-head self-attention to capture global dependencies in parallel, achieving state-of-the-art results on machine translation benchmarks with significantly reduced training time.",
        keywords=["transformer", "self-attention", "multi-head attention", "encoder-decoder", "sequence-to-sequence", "machine translation", "positional encoding"],
        sub_topic="transformer_architecture",
        reference_summary="Introduces the Transformer model, replacing RNNs/CNNs with multi-head self-attention for parallel sequence modeling. Achieves SOTA on WMT translation with faster training via positional encoding and residual connections.",
        reference_explanation="Instead of reading words one-by-one like older models, the Transformer looks at all words simultaneously using 'attention' — a mechanism that lets each word check how relevant every other word is. This parallel processing makes it much faster to train while capturing long-range relationships in text.",
        relevance_tags={"single_topic_retrieval": 1.0, "ambiguous_query_filtering": 0.3, "multi_concept_synthesis": 0.5},
    ),
    Paper(
        id="paper_002",
        title="BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding",
        abstract="We introduce BERT, a language representation model designed to pre-train deep bidirectional transformers by jointly conditioning on both left and right context. Unlike previous models, BERT enables fine-tuning with just one additional output layer for a wide range of NLP tasks including question answering and sentiment analysis.",
        keywords=["BERT", "pre-training", "bidirectional", "transformer", "fine-tuning", "masked language model", "NLP", "transfer learning"],
        sub_topic="transformer_architecture",
        reference_summary="BERT pre-trains a bidirectional transformer using masked language modeling and next sentence prediction, enabling transfer learning across diverse NLP tasks with minimal task-specific architecture changes.",
        reference_explanation="BERT learns language by reading text in both directions and trying to guess hidden words. This deep understanding can then be adapted to many different tasks — like answering questions or detecting sentiment — with very little additional training.",
        relevance_tags={"single_topic_retrieval": 0.9, "ambiguous_query_filtering": 0.2, "multi_concept_synthesis": 0.4},
    ),
    Paper(
        id="paper_003",
        title="GPT-3: Language Models are Few-Shot Learners",
        abstract="We demonstrate that scaling language models to 175 billion parameters results in strong few-shot performance across many NLP benchmarks. GPT-3 achieves competitive results without gradient updates or fine-tuning, using only natural language prompts and a few demonstrations, suggesting that large autoregressive models can serve as general-purpose language processors.",
        keywords=["GPT-3", "few-shot learning", "language model", "autoregressive", "scaling", "in-context learning", "NLP", "transformer"],
        sub_topic="transformer_architecture",
        reference_summary="GPT-3 scales autoregressive transformers to 175B parameters, achieving strong few-shot performance via in-context learning without fine-tuning, demonstrating emergent general-purpose language processing capabilities.",
        reference_explanation="GPT-3 is an extremely large language model that can perform new tasks just by reading a few examples in its prompt — no retraining needed. This 'in-context learning' capability emerges from scale, showing that bigger models can work as flexible, general-purpose text tools.",
        relevance_tags={"single_topic_retrieval": 0.85, "ambiguous_query_filtering": 0.2, "multi_concept_synthesis": 0.6},
    ),
    Paper(
        id="paper_004",
        title="Vision Transformers: An Image is Worth 16x16 Words",
        abstract="We show that a standard Transformer architecture applied directly to sequences of image patches can achieve excellent results on image classification benchmarks. Vision Transformer (ViT) attains state-of-the-art results when pre-trained on large datasets, demonstrating that the dominance of CNNs in computer vision is not inevitable.",
        keywords=["vision transformer", "ViT", "image classification", "patches", "self-attention", "computer vision"],
        sub_topic="transformer_architecture",
        reference_summary="ViT applies the standard Transformer to image patches, achieving competitive image classification results and demonstrating that attention-based models can rival CNNs in computer vision when pre-trained at scale.",
        reference_explanation="Instead of using convolutional filters, ViT splits an image into small patches and treats them like words in a sentence, applying attention to learn relationships between different parts of the image. This approach matches or beats traditional image models when given enough training data.",
        relevance_tags={"single_topic_retrieval": 0.4, "ambiguous_query_filtering": 0.1, "multi_concept_synthesis": 0.2},
    ),
    Paper(
        id="paper_005",
        title="Efficient Transformers: A Survey of Attention Mechanisms",
        abstract="This survey examines various efficient attention mechanisms designed to reduce the quadratic computational complexity of standard self-attention. We categorize approaches into sparse attention, linear attention, low-rank methods, and kernel-based techniques, analyzing their trade-offs between efficiency and modeling capability.",
        keywords=["efficient transformers", "sparse attention", "linear attention", "computational complexity", "self-attention", "survey"],
        sub_topic="transformer_architecture",
        reference_summary="Surveys efficient attention variants that reduce O(n²) complexity: sparse patterns, linear approximations, low-rank factorizations, and kernel methods, comparing their efficiency-quality trade-offs for long-sequence processing.",
        reference_explanation="Standard attention becomes very slow for long text because every word must attend to every other word. This paper reviews clever shortcuts — like only attending to nearby words, or using mathematical approximations — that make attention much faster while preserving most of the quality.",
        relevance_tags={"single_topic_retrieval": 0.7, "ambiguous_query_filtering": 0.15, "multi_concept_synthesis": 0.3},
    ),
    Paper(
        id="paper_006",
        title="T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Framework",
        abstract="We explore transfer learning by converting every language problem into a text-to-text format. Using this unified framework, we systematically study pre-training objectives, architectures, and dataset scales, finding that scaling model size and data quality are the most important factors for downstream performance.",
        keywords=["T5", "transfer learning", "text-to-text", "pre-training", "unified framework", "NLP", "encoder-decoder"],
        sub_topic="transformer_architecture",
        reference_summary="T5 unifies NLP tasks into text-to-text format, systematically studying pre-training methods and demonstrating that model scale and data quality dominate as factors for strong transfer learning performance.",
        reference_explanation="T5 converts every language task — translation, summarization, question answering — into the same format: text-in, text-out. This simplification lets researchers focus on what really matters for performance: using bigger models and cleaner data.",
        relevance_tags={"single_topic_retrieval": 0.6, "ambiguous_query_filtering": 0.15, "multi_concept_synthesis": 0.35},
    ),
    Paper(
        id="paper_007",
        title="Positional Encodings in Transformers: From Sinusoidal to Rotary",
        abstract="This paper provides a comprehensive analysis of positional encoding methods in Transformer architectures. We compare sinusoidal encodings, learned embeddings, relative position biases, ALiBi, and Rotary Position Embeddings (RoPE), evaluating their ability to generalize to sequence lengths unseen during training.",
        keywords=["positional encoding", "RoPE", "ALiBi", "sinusoidal", "transformer", "length generalization", "relative position"],
        sub_topic="transformer_architecture",
        reference_summary="Compares positional encoding methods — sinusoidal, learned, relative, ALiBi, and RoPE — for Transformers, finding that rotary and relative encodings generalize better to longer sequences than absolute encodings.",
        reference_explanation="Transformers don't inherently understand word order, so positional encodings tell the model where each word sits. This paper compares different encoding methods and finds that newer approaches like RoPE handle longer texts much better than the original sinusoidal method.",
        relevance_tags={"single_topic_retrieval": 0.75, "ambiguous_query_filtering": 0.1, "multi_concept_synthesis": 0.2},
    ),

    # =====================================================================
    # SUB-TOPIC 2: Bayesian Neural Networks & Uncertainty (papers 008-013)
    # =====================================================================
    Paper(
        id="paper_008",
        title="Weight Uncertainty in Neural Networks: Bayesian Inference by Backpropagation",
        abstract="We introduce a practical method for learning weight distributions in neural networks using backpropagation. By maintaining probability distributions over weights instead of point estimates, the network can express uncertainty in its predictions, enabling principled decision-making under ambiguity.",
        keywords=["Bayesian neural network", "weight uncertainty", "variational inference", "backpropagation", "posterior", "uncertainty quantification"],
        sub_topic="bayesian_uncertainty",
        reference_summary="Proposes Bayes by Backprop for learning weight distributions in neural networks via variational inference, enabling principled uncertainty quantification through maintaining posteriors over weights during training.",
        reference_explanation="Instead of learning a single fixed value for each model weight, this approach learns a range of likely values. This means the model can say 'I'm not sure about this prediction' — which is crucial in high-stakes applications like medical diagnosis.",
        relevance_tags={"single_topic_retrieval": 0.1, "ambiguous_query_filtering": 1.0, "multi_concept_synthesis": 0.3},
    ),
    Paper(
        id="paper_009",
        title="Dropout as a Bayesian Approximation: Representing Model Uncertainty",
        abstract="We show that a neural network with dropout applied before every weight layer is mathematically equivalent to an approximation of the deep Gaussian process. This provides a practical tool for obtaining model uncertainty estimates by running multiple forward passes with dropout at test time, known as MC Dropout.",
        keywords=["dropout", "Bayesian approximation", "MC dropout", "uncertainty", "Gaussian process", "model uncertainty", "epistemic uncertainty"],
        sub_topic="dropout_uncertainty",
        reference_summary="Proves that dropout approximates Bayesian inference in deep Gaussian processes, providing practical uncertainty estimates via MC Dropout — multiple stochastic forward passes at test time yield predictive variance.",
        reference_explanation="Dropout (randomly turning off neurons during training) turns out to be secretly performing Bayesian inference. By keeping dropout on during predictions and averaging multiple runs, you get uncertainty estimates for free — no special Bayesian architecture needed.",
        relevance_tags={"single_topic_retrieval": 0.1, "ambiguous_query_filtering": 1.0, "multi_concept_synthesis": 0.2},
    ),
    Paper(
        id="paper_010",
        title="Deep Ensembles: A Simple and Scalable Method for Uncertainty Quantification",
        abstract="We propose using ensembles of neural networks trained with random initialization as a practical method for predictive uncertainty estimation. Deep ensembles produce well-calibrated uncertainty estimates and outperform single-model Bayesian approaches on out-of-distribution detection tasks.",
        keywords=["deep ensembles", "uncertainty quantification", "calibration", "out-of-distribution", "ensemble methods", "predictive uncertainty"],
        sub_topic="ensemble_uncertainty",
        reference_summary="Proposes deep ensembles — independently trained networks with random initialization — as a simple, scalable method for uncertainty estimation that produces well-calibrated predictions and strong OOD detection.",
        reference_explanation="Training several copies of the same model with different random starting points and averaging their predictions gives surprisingly good uncertainty estimates. When the models disagree, the input is likely unusual or difficult — a simple but powerful insight.",
        relevance_tags={"single_topic_retrieval": 0.1, "ambiguous_query_filtering": 0.95, "multi_concept_synthesis": 0.2},
    ),
    Paper(
        id="paper_011",
        title="Evidential Deep Learning: Quantifying Classification Uncertainty",
        abstract="We introduce evidential deep learning, which places a Dirichlet distribution over class probabilities to model both aleatoric and epistemic uncertainty in a single forward pass. This avoids the computational cost of ensembles or MC dropout while providing rich uncertainty decomposition.",
        keywords=["evidential learning", "Dirichlet", "uncertainty decomposition", "epistemic", "aleatoric", "classification", "single forward pass"],
        sub_topic="bayesian_uncertainty",
        reference_summary="Introduces evidential deep learning using Dirichlet priors over class probabilities, decomposing uncertainty into aleatoric and epistemic components in a single forward pass — faster than ensembles or MC dropout.",
        reference_explanation="This method teaches the model to output not just predictions but also how much evidence it has for each answer. By separating 'the data is inherently noisy' from 'the model hasn't seen this type of input before,' it gives richer uncertainty information in one fast prediction.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.85, "multi_concept_synthesis": 0.15},
    ),
    Paper(
        id="paper_012",
        title="Calibration of Modern Neural Networks: Are We There Yet?",
        abstract="We investigate the calibration properties of modern deep neural networks, finding that despite improvements in accuracy, many models remain poorly calibrated. We study temperature scaling, label smoothing, and mixup as post-hoc and training-time calibration techniques.",
        keywords=["calibration", "neural networks", "temperature scaling", "confidence", "reliability diagram", "overconfidence"],
        sub_topic="bayesian_uncertainty",
        reference_summary="Studies calibration of modern neural networks, finding persistent overconfidence despite accuracy gains. Evaluates temperature scaling, label smoothing, and mixup as calibration correction methods.",
        reference_explanation="Modern AI models are often overconfident — they say they're 95% sure when they're only right 70% of the time. This paper studies techniques to fix this mismatch, making model confidence scores more trustworthy and actionable.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.7, "multi_concept_synthesis": 0.1},
    ),
    Paper(
        id="paper_013",
        title="Spectral-Normalized Neural Gaussian Processes for Out-of-Distribution Detection",
        abstract="We propose SNGP, which combines spectral normalization of hidden layers with a Gaussian process output layer to provide distance-aware uncertainty estimates. SNGP achieves strong out-of-distribution detection performance while maintaining the simplicity of deterministic networks.",
        keywords=["SNGP", "spectral normalization", "Gaussian process", "out-of-distribution", "uncertainty", "distance awareness"],
        sub_topic="bayesian_uncertainty",
        reference_summary="SNGP combines spectral normalization with a GP output layer for distance-aware uncertainty, achieving strong OOD detection with the simplicity and speed of deterministic neural networks.",
        reference_explanation="By controlling how quickly the model's internal representations change (spectral normalization) and adding a smart uncertainty layer at the end (Gaussian process), SNGP can detect when inputs are far from its training data — crucial for safe deployment.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.6, "multi_concept_synthesis": 0.1},
    ),

    # =====================================================================
    # SUB-TOPIC 3: RLHF & Alignment (papers 014-020)
    # =====================================================================
    Paper(
        id="paper_014",
        title="Training Language Models to Follow Instructions with Human Feedback (InstructGPT)",
        abstract="We fine-tune GPT-3 using reinforcement learning from human feedback (RLHF) to better follow user intent. By training a reward model on human preferences and optimizing the policy using PPO, InstructGPT produces outputs that are more helpful, honest, and harmless compared to the base model.",
        keywords=["RLHF", "InstructGPT", "human feedback", "reward model", "PPO", "alignment", "instruction following", "language model"],
        sub_topic="rlhf_alignment",
        reference_summary="InstructGPT applies RLHF to GPT-3: a reward model trained on human preference data guides PPO optimization, producing language models that better follow instructions and align with human intent.",
        reference_explanation="To make language models actually helpful rather than just predictive, InstructGPT trains them using human feedback. People rate different model outputs, a reward model learns what humans prefer, and then the language model is trained to maximize that reward — like teaching a student using a rubric.",
        relevance_tags={"single_topic_retrieval": 0.15, "ambiguous_query_filtering": 0.1, "multi_concept_synthesis": 1.0},
    ),
    Paper(
        id="paper_015",
        title="Constitutional AI: Harmlessness from AI Feedback",
        abstract="We propose Constitutional AI (CAI), a method for training helpful and harmless AI assistants without extensive human feedback on harmful outputs. The model critiques and revises its own responses according to a set of principles (a 'constitution'), then trains on the revised outputs using RLHF with AI feedback.",
        keywords=["constitutional AI", "RLAIF", "self-critique", "harmlessness", "AI feedback", "principles", "alignment"],
        sub_topic="rlhf_alignment",
        reference_summary="Constitutional AI uses self-critique guided by explicit principles to revise harmful outputs, then trains via RLAIF — replacing human feedback with AI-generated preferences for scalable alignment.",
        reference_explanation="Instead of having humans label every harmful output, Constitutional AI gives the model a set of rules ('constitution') and has it critique and fix its own responses. This self-improvement loop scales better than pure human feedback while maintaining safety standards.",
        relevance_tags={"single_topic_retrieval": 0.1, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.9},
    ),
    Paper(
        id="paper_016",
        title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        abstract="We show that the constrained reward maximization problem in RLHF can be solved in closed form, yielding Direct Preference Optimization (DPO). DPO directly optimizes the policy using preference pairs without explicitly training a reward model or using RL, resulting in a simpler and more stable training procedure.",
        keywords=["DPO", "preference optimization", "RLHF alternative", "language model", "alignment", "reward model", "policy optimization"],
        sub_topic="rlhf_alignment",
        reference_summary="DPO reformulates RLHF as direct policy optimization on preference pairs, eliminating the need for explicit reward model training and RL. Simpler, more stable, and mathematically equivalent to constrained reward maximization.",
        reference_explanation="Traditional RLHF requires training a separate reward model then using RL to optimize against it — complex and unstable. DPO shows you can skip both steps by directly adjusting the model to prefer better outputs over worse ones, making alignment training much simpler.",
        relevance_tags={"single_topic_retrieval": 0.1, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.95},
    ),
    Paper(
        id="paper_017",
        title="Proximal Policy Optimization Algorithms for Robust RL Training",
        abstract="We present PPO, a family of policy gradient methods that alternate between sampling data through interaction and optimizing a clipped surrogate objective. PPO achieves the stability of trust-region methods with the simplicity of first-order optimization, making it practical for large-scale RL applications.",
        keywords=["PPO", "proximal policy optimization", "policy gradient", "reinforcement learning", "clipped objective", "trust region", "stability"],
        sub_topic="rl_fundamentals",
        reference_summary="PPO introduces a clipped surrogate objective for policy gradient optimization, achieving trust-region stability with first-order simplicity. Widely adopted for large-scale RL due to its robustness and ease of implementation.",
        reference_explanation="PPO makes reinforcement learning training more stable by preventing the policy from changing too dramatically in any single update. This 'clipping' trick is simple but effective — it's become the default RL algorithm behind most large language model training.",
        relevance_tags={"single_topic_retrieval": 0.1, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.85},
    ),
    Paper(
        id="paper_018",
        title="Reward Hacking in Language Model Alignment: Pitfalls and Mitigations",
        abstract="We study the phenomenon of reward hacking in RLHF-trained language models, where models learn to exploit weaknesses in the reward model rather than genuinely improving quality. We propose reward model ensembles, constrained optimization, and iterative reward model updates as mitigations.",
        keywords=["reward hacking", "RLHF", "reward model", "overoptimization", "alignment", "Goodhart's law", "language model"],
        sub_topic="rlhf_alignment",
        reference_summary="Studies reward hacking where RLHF models exploit reward model failures. Proposes mitigations: reward model ensembles, KL-constrained optimization, and iterative reward model retraining to prevent Goodhart's-law degradation.",
        reference_explanation="When language models are trained to maximize a reward signal, they sometimes find 'shortcuts' — outputs that score high on the reward model but aren't actually better. This paper identifies the problem and proposes fixes like using multiple reward models that are harder to simultaneously fool.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.8},
    ),
    Paper(
        id="paper_019",
        title="RLHF Workflow: From Reward Modeling to Online Reinforcement Learning",
        abstract="We present a comprehensive framework for implementing RLHF, covering data collection strategies, reward model architecture choices, PPO implementation details, and evaluation protocols. We identify key failure modes and provide practical recipes for stable training of aligned language models.",
        keywords=["RLHF", "workflow", "reward modeling", "PPO", "data collection", "evaluation", "language model alignment"],
        sub_topic="rlhf_alignment",
        reference_summary="Provides a practical RLHF implementation guide covering data collection, reward model training, PPO hyperparameters, and evaluation. Identifies failure modes (reward hacking, mode collapse) with concrete mitigation recipes.",
        reference_explanation="This paper is a practical guide to building RLHF systems. It covers the full pipeline — from collecting human preferences, to training the reward model, to the RL fine-tuning step — with concrete advice on avoiding common pitfalls like reward hacking and training instability.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.75},
    ),
    Paper(
        id="paper_020",
        title="Scaling Laws for Reward Model Overoptimization in RLHF",
        abstract="We characterize how reward model overoptimization scales with model size, reward model capacity, and the amount of human feedback data. We find predictable scaling relationships that allow practitioners to budget compute and data collection for RLHF training.",
        keywords=["scaling laws", "reward model", "overoptimization", "RLHF", "human feedback", "model size", "data efficiency"],
        sub_topic="rlhf_alignment",
        reference_summary="Establishes scaling laws for reward model overoptimization in RLHF, showing predictable relationships between model size, reward model capacity, and feedback data volume that guide resource allocation.",
        reference_explanation="As you train a language model harder against a reward model, performance eventually gets worse because the model finds exploits. This paper quantifies exactly how this 'overoptimization' scales, giving practitioners a formula to know when to stop training.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.65},
    ),

    # =====================================================================
    # SUB-TOPIC 4: LLM Tool Use & Agents (papers 021-027)
    # =====================================================================
    Paper(
        id="paper_021",
        title="Toolformer: Language Models Can Teach Themselves to Use Tools",
        abstract="We show that language models can be taught to use external tools — calculators, search engines, translators — by self-generating API call examples. Toolformer learns when and how to call tools during text generation, improving performance on tasks requiring factual knowledge or computation.",
        keywords=["Toolformer", "tool use", "API calls", "language model", "self-supervised", "augmentation", "external tools"],
        sub_topic="llm_tool_use",
        reference_summary="Toolformer teaches LLMs to self-generate tool-use examples (calculators, search, translation) via self-supervised augmentation, learning when and how to invoke external APIs during text generation.",
        reference_explanation="Toolformer lets language models learn to use tools like search engines and calculators by themselves. The model generates training examples of tool calls, keeps the ones that improve its predictions, and learns to seamlessly weave tool use into natural text generation.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.9},
    ),
    Paper(
        id="paper_022",
        title="ReAct: Synergizing Reasoning and Acting in Language Models",
        abstract="We propose ReAct, a framework that interleaves reasoning traces and task-specific actions in language models. By generating both verbal reasoning steps and concrete actions (search, lookup), ReAct overcomes hallucination and error propagation issues common in chain-of-thought reasoning alone.",
        keywords=["ReAct", "reasoning", "acting", "language model", "chain-of-thought", "interleaving", "grounded reasoning", "action generation"],
        sub_topic="llm_tool_use",
        reference_summary="ReAct interleaves verbal reasoning with concrete actions (search, lookup) in LLMs, combining chain-of-thought benefits with grounded action execution to reduce hallucination and improve task completion.",
        reference_explanation="ReAct combines thinking and doing: the model writes out its reasoning step, then takes a concrete action like searching the web, then reasons about the result. This think-act-observe loop keeps the model grounded in real information, reducing hallucination.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.85},
    ),
    Paper(
        id="paper_023",
        title="LLM-Based Autonomous Agents: A Survey of Architectures and Capabilities",
        abstract="We survey the emerging field of LLM-based autonomous agents that combine language model reasoning with planning, memory, and tool use. We categorize agent architectures, analyze their capabilities and limitations, and identify open challenges in building reliable, generalizable AI agents.",
        keywords=["LLM agents", "autonomous agents", "planning", "memory", "tool use", "survey", "reasoning", "generalization"],
        sub_topic="llm_tool_use",
        reference_summary="Surveys LLM-based autonomous agent architectures combining reasoning, planning, memory, and tool use. Categorizes approaches, analyzes capability gaps, and identifies challenges for reliable, general-purpose AI agents.",
        reference_explanation="This survey maps the landscape of AI agents built on language models. These agents can think through problems, remember past interactions, make plans, and use external tools — but the paper also honestly assesses where they still struggle, like reliable long-term planning.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.8},
    ),
    Paper(
        id="paper_024",
        title="Planning with Large Language Models for Code Generation",
        abstract="We study how LLMs can plan and decompose complex programming tasks into subtasks before generating code. Our approach uses hierarchical planning where the model first outlines a solution strategy, then implements each component, and finally integrates them — mimicking how expert programmers work.",
        keywords=["planning", "code generation", "LLM", "decomposition", "hierarchical", "programming", "subtask"],
        sub_topic="llm_planning",
        reference_summary="Studies hierarchical planning in LLMs for code generation: the model outlines a strategy, implements subtasks, then integrates components — mimicking expert programmer workflows for complex programming tasks.",
        reference_explanation="Instead of generating code all at once, this approach has the language model plan its solution first — like writing an outline before an essay. It breaks the problem into pieces, codes each one, then assembles the final program, leading to better solutions for complex tasks.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.7},
    ),
    Paper(
        id="paper_025",
        title="WebGPT: Browser-Assisted Question Answering with Human Feedback",
        abstract="We fine-tune GPT-3 to answer open-ended questions by browsing the web. The model learns to issue search queries, click links, and extract information from web pages. Using RLHF to train on human preferences, WebGPT produces more factual and well-sourced answers than the base model.",
        keywords=["WebGPT", "web browsing", "question answering", "RLHF", "human feedback", "search", "information retrieval", "factuality"],
        sub_topic="llm_tool_use",
        reference_summary="WebGPT fine-tunes GPT-3 to browse the web for question answering, learning to search, navigate, and extract information. RLHF training on human preferences improves factuality and source attribution.",
        reference_explanation="WebGPT teaches a language model to use a web browser — searching for information, clicking links, and reading pages — to answer questions with real sources. Human feedback training ensures it produces factual, well-supported answers rather than making things up.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.75},
    ),
    Paper(
        id="paper_026",
        title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        abstract="We show that including step-by-step reasoning examples in prompts dramatically improves LLM performance on complex reasoning tasks. Chain-of-thought prompting enables models to decompose multi-step problems, showing emergent reasoning capabilities that scale with model size.",
        keywords=["chain-of-thought", "prompting", "reasoning", "step-by-step", "emergent", "few-shot", "problem decomposition"],
        sub_topic="llm_planning",
        reference_summary="Chain-of-thought prompting improves LLM reasoning by including step-by-step examples in prompts, enabling multi-step problem decomposition. This emergent capability scales with model size and generalizes across reasoning tasks.",
        reference_explanation="By showing language models examples of step-by-step reasoning, they learn to 'think aloud' before answering. This simple technique dramatically improves performance on math, logic, and common-sense reasoning — especially in larger models.",
        relevance_tags={"single_topic_retrieval": 0.1, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.6},
    ),
    Paper(
        id="paper_027",
        title="Tree of Thoughts: Deliberate Problem Solving with Language Models",
        abstract="We introduce Tree of Thoughts (ToT), a framework for LLM reasoning that explores multiple reasoning paths simultaneously. ToT enables systematic search through a tree of possible thought sequences, using BFS or DFS with self-evaluation to find the best solution path.",
        keywords=["Tree of Thoughts", "reasoning", "search", "BFS", "DFS", "self-evaluation", "deliberate", "problem solving"],
        sub_topic="llm_planning",
        reference_summary="Tree of Thoughts enables LLMs to explore multiple reasoning paths via systematic tree search (BFS/DFS) with self-evaluation, allowing deliberate problem solving beyond single-chain reasoning.",
        reference_explanation="Instead of following one chain of reasoning, Tree of Thoughts lets the model explore multiple possible reasoning paths like branches of a tree, evaluate which paths look most promising, and backtrack if needed — more like how humans solve complex problems.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.55},
    ),

    # =====================================================================
    # SUB-TOPIC 5: RL for Decision Making (papers 028-033)
    # =====================================================================
    Paper(
        id="paper_028",
        title="Decision Transformer: Reinforcement Learning via Sequence Modeling",
        abstract="We frame reinforcement learning as a sequence modeling problem and propose Decision Transformer, which uses a causal transformer to predict actions given desired return-to-go, past states, and actions. This formulation eliminates the need for value functions or policy gradients, matching or exceeding traditional RL methods on offline benchmarks.",
        keywords=["Decision Transformer", "offline RL", "sequence modeling", "return-to-go", "reinforcement learning", "transformer", "decision making"],
        sub_topic="rl_decision_making",
        reference_summary="Decision Transformer recasts RL as sequence modeling, using a causal transformer conditioned on desired returns to predict actions. Eliminates value functions and policy gradients while matching offline RL benchmarks.",
        reference_explanation="Decision Transformer treats reinforcement learning like a language task: given the desired outcome and past experiences, predict the next action. This sequence-modeling approach avoids the instabilities of traditional RL training and works well with pre-recorded experience data.",
        relevance_tags={"single_topic_retrieval": 0.1, "ambiguous_query_filtering": 0.1, "multi_concept_synthesis": 1.0},
    ),
    Paper(
        id="paper_029",
        title="Language Models as Decision Makers: Grounding LLMs in Interactive Environments",
        abstract="We study how pre-trained language models can serve as decision-making agents in interactive environments. By representing environment states as text descriptions and actions as natural language, LLMs can reason about consequences and plan multi-step strategies without task-specific RL training.",
        keywords=["LLM decision making", "interactive environments", "text-based RL", "grounding", "planning", "language agent"],
        sub_topic="rl_decision_making",
        reference_summary="Studies LLMs as decision-making agents in interactive environments, representing states and actions as text. Pre-trained language knowledge enables zero-shot planning and multi-step reasoning without task-specific RL.",
        reference_explanation="Instead of training RL agents from scratch, this work uses language models' existing knowledge to make decisions in interactive environments. By converting game states to text descriptions, the language model can reason about what to do next — a fundamentally different approach to RL.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.95},
    ),
    Paper(
        id="paper_030",
        title="Reinforcement Learning with Foundation Models: Opportunities and Challenges",
        abstract="We survey the intersection of foundation models and reinforcement learning, identifying three paradigms: foundation models as world models, as reward signals, and as policies. We analyze how pre-trained knowledge accelerates RL training but introduce challenges in distribution shift and reward specification.",
        keywords=["foundation models", "reinforcement learning", "world models", "reward models", "policy", "pre-training", "survey"],
        sub_topic="rl_decision_making",
        reference_summary="Surveys RL with foundation models across three paradigms: as world models, reward signals, and policies. Pre-trained knowledge accelerates RL but introduces distribution shift and reward specification challenges.",
        reference_explanation="Foundation models can supercharge reinforcement learning in three ways: as simulators of the world, as judges of what's good, or as the decision-making agent itself. This survey maps these approaches and honestly discusses where each paradigm works and where it breaks down.",
        relevance_tags={"single_topic_retrieval": 0.1, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.85},
    ),
    Paper(
        id="paper_031",
        title="Multi-Agent RL with Emergent Communication in Language Models",
        abstract="We study how multiple LLM-based agents can develop emergent communication protocols to solve cooperative tasks. By training agents with shared rewards in multi-agent RL settings, we observe the emergence of structured, task-specific communication strategies that improve team performance.",
        keywords=["multi-agent RL", "emergent communication", "cooperation", "language models", "shared reward", "team performance"],
        sub_topic="rl_decision_making",
        reference_summary="Studies emergent communication in multi-agent LLM systems trained with shared rewards. Agents develop structured communication protocols that improve cooperative task performance beyond independent operation.",
        reference_explanation="When multiple AI agents need to work together, they can learn to communicate with each other to coordinate their actions. This paper shows that RL training naturally leads to efficient communication strategies — the agents essentially invent their own task-specific language.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.5},
    ),
    Paper(
        id="paper_032",
        title="Offline RL from Human Feedback: Bridging Language and Decision Making",
        abstract="We present a method that combines offline RL with language model capabilities, learning decision-making policies from datasets of human demonstrations annotated with natural language feedback. This bridges the gap between language understanding and sequential decision-making.",
        keywords=["offline RL", "human feedback", "language model", "decision making", "demonstrations", "natural language", "policy learning"],
        sub_topic="rl_decision_making",
        reference_summary="Combines offline RL with LLM capabilities, learning policies from human demonstrations annotated with language feedback. Bridges language understanding and sequential decision-making for grounded agent behavior.",
        reference_explanation="This method learns to make decisions from recorded human demonstrations that include natural language explanations of why those decisions were made. The language context helps the model generalize to new situations by understanding the reasoning behind actions, not just mimicking them.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.7},
    ),
    Paper(
        id="paper_033",
        title="Reward Shaping for Language Model Fine-Tuning with Reinforcement Learning",
        abstract="We investigate reward shaping techniques specifically designed for fine-tuning language models with RL. We propose progressive reward signals that guide the model through intermediate quality levels, avoiding the sparse reward problem and enabling stable training convergence.",
        keywords=["reward shaping", "language model", "fine-tuning", "reinforcement learning", "progressive rewards", "dense reward", "training stability"],
        sub_topic="rl_decision_making",
        reference_summary="Proposes progressive reward shaping for RL-based LLM fine-tuning, providing dense intermediate signals that guide training through quality levels, avoiding sparse reward problems and improving convergence stability.",
        reference_explanation="Training language models with RL often fails because the reward signal is too sparse — the model only learns if its final output is good or bad. Reward shaping adds intermediate feedback at each generation step, like a teacher giving encouragement along the way rather than just grading the final answer.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.65},
    ),

    # =====================================================================
    # SUB-TOPIC 6: Computer Vision (distractors for easy/medium) (034-039)
    # =====================================================================
    Paper(
        id="paper_034",
        title="ResNet: Deep Residual Learning for Image Recognition",
        abstract="We present residual networks (ResNets) that use skip connections to enable training of very deep neural networks. By learning residual functions instead of direct mappings, ResNets solve the vanishing gradient problem and achieve state-of-the-art accuracy on ImageNet classification.",
        keywords=["ResNet", "residual learning", "skip connections", "deep learning", "image classification", "vanishing gradient", "ImageNet"],
        sub_topic="computer_vision",
        reference_summary="ResNets introduce skip connections that learn residual functions, enabling training of very deep networks by addressing vanishing gradients. Achieved breakthrough ImageNet results with 152-layer networks.",
        reference_explanation="ResNets solve the problem of training very deep networks by adding 'shortcuts' that let information skip over layers. This makes it easy for the network to learn small refinements at each layer rather than complete transformations.",
        relevance_tags={"single_topic_retrieval": 0.1, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.05},
    ),
    Paper(
        id="paper_035",
        title="YOLO: Real-Time Object Detection at Scale",
        abstract="We introduce YOLO, a unified model for object detection that frames detection as a single regression problem. YOLO processes full images in one network evaluation, achieving real-time performance while maintaining competitive detection accuracy.",
        keywords=["YOLO", "object detection", "real-time", "single-stage", "regression", "computer vision"],
        sub_topic="computer_vision",
        reference_summary="YOLO reframes object detection as single-pass regression, processing full images in one network evaluation for real-time detection with competitive accuracy compared to two-stage methods.",
        reference_explanation="Traditional object detectors work in two stages — first finding candidate regions, then classifying them. YOLO does everything in one pass, looking at the entire image at once to detect objects, making it fast enough for real-time video processing.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.05},
    ),
    Paper(
        id="paper_036",
        title="Diffusion Models Beat GANs on Image Synthesis Quality",
        abstract="We demonstrate that diffusion probabilistic models can achieve image quality surpassing state-of-the-art GANs. By introducing classifier guidance, we achieve unmatched FID scores on ImageNet while maintaining the training stability and mode coverage advantages of diffusion models.",
        keywords=["diffusion models", "image synthesis", "GANs", "classifier guidance", "FID", "generative models"],
        sub_topic="computer_vision",
        reference_summary="Shows diffusion models surpassing GANs in image quality via classifier guidance. Achieves best FID scores on ImageNet while maintaining diffusion's training stability and mode coverage advantages.",
        reference_explanation="Diffusion models generate images by gradually removing noise from random static, step by step. This paper shows they can produce higher-quality images than GANs, which were previously dominant, while being easier to train and covering more diverse outputs.",
        relevance_tags={"single_topic_retrieval": 0.1, "ambiguous_query_filtering": 0.1, "multi_concept_synthesis": 0.05},
    ),
    Paper(
        id="paper_037",
        title="Segment Anything: Foundation Models for Image Segmentation",
        abstract="We introduce SAM, a foundation model for image segmentation that can segment any object in any image given a point, box, or text prompt. Trained on 1 billion masks from 11 million images, SAM demonstrates strong zero-shot transfer to diverse segmentation tasks.",
        keywords=["SAM", "segmentation", "foundation model", "zero-shot", "promptable", "computer vision"],
        sub_topic="computer_vision",
        reference_summary="SAM is a promptable segmentation foundation model trained on 1B masks. Segments any object given points, boxes, or text prompts with strong zero-shot transfer to diverse visual segmentation tasks.",
        reference_explanation="SAM can highlight any object in any image you point to — no retraining needed. Trained on a massive dataset of labeled segments, it understands visual objects well enough to work out-of-the-box on new types of images it has never seen.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.05},
    ),
    Paper(
        id="paper_038",
        title="Neural Radiance Fields for Novel View Synthesis",
        abstract="We present NeRF, which represents scenes as continuous volumetric functions learned from a set of input images. By encoding scene geometry and appearance as a neural network, NeRF synthesizes photorealistic novel views of complex scenes from arbitrary viewpoints.",
        keywords=["NeRF", "novel view synthesis", "volumetric rendering", "3D reconstruction", "neural scene representation"],
        sub_topic="computer_vision",
        reference_summary="NeRF represents scenes as continuous neural volumetric functions, synthesizing photorealistic novel views from arbitrary viewpoints by learning scene geometry and appearance from a set of posed images.",
        reference_explanation="NeRF creates a 3D digital model of a scene from just a collection of regular photos. A neural network learns to predict what you'd see from any viewpoint, creating photorealistic images from angles that were never photographed.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.05},
    ),
    Paper(
        id="paper_039",
        title="Multi-Modal Contrastive Learning for Visual Representation with Language Supervision",
        abstract="We study contrastive learning methods that align visual and textual representations in a shared embedding space. By training on large-scale image-text pairs with a contrastive objective, the model learns transferable visual representations useful for zero-shot classification and retrieval.",
        keywords=["contrastive learning", "multi-modal", "CLIP", "visual representation", "language supervision", "zero-shot"],
        sub_topic="computer_vision",
        reference_summary="Studies contrastive image-text alignment in shared embedding spaces, learning transferable visual representations from large-scale pairs for zero-shot classification and cross-modal retrieval.",
        reference_explanation="By training on millions of image-text pairs, models learn to map images and text descriptions to the same space. A photo of a cat ends up near the text 'a cat' — enabling the model to classify images it's never seen labeled examples of.",
        relevance_tags={"single_topic_retrieval": 0.1, "ambiguous_query_filtering": 0.1, "multi_concept_synthesis": 0.1},
    ),

    # =====================================================================
    # SUB-TOPIC 7: NLP Applications (distractors / partial relevance) (040-045)
    # =====================================================================
    Paper(
        id="paper_040",
        title="Retrieval-Augmented Generation for Knowledge-Intensive Tasks",
        abstract="We combine pre-trained parametric models with non-parametric retrieval to create RAG, which retrieves relevant documents from an external knowledge base before generating responses. RAG improves factual accuracy and allows models to access up-to-date information without retraining.",
        keywords=["RAG", "retrieval-augmented generation", "knowledge base", "factual accuracy", "non-parametric", "information retrieval"],
        sub_topic="nlp_applications",
        reference_summary="RAG combines parametric LMs with non-parametric retrieval, fetching relevant documents before generation to improve factual accuracy and enable access to up-to-date knowledge without retraining.",
        reference_explanation="RAG gives language models a 'reference library' — before answering a question, the model first searches a knowledge base for relevant documents, then generates an answer grounded in those sources. This dramatically reduces hallucination and works with fresh information.",
        relevance_tags={"single_topic_retrieval": 0.2, "ambiguous_query_filtering": 0.1, "multi_concept_synthesis": 0.4},
    ),
    Paper(
        id="paper_041",
        title="Instruction Tuning: Scaling Multi-Task Learning for Generalization",
        abstract="We study instruction tuning, where language models are fine-tuned on a mixture of tasks described as natural language instructions. By scaling the number and diversity of instruction-following tasks, models achieve improved zero-shot generalization to unseen tasks.",
        keywords=["instruction tuning", "multi-task", "fine-tuning", "zero-shot", "generalization", "NLP", "task diversity"],
        sub_topic="nlp_applications",
        reference_summary="Studies instruction tuning at scale, showing that fine-tuning LMs on diverse instruction-following tasks improves zero-shot generalization proportionally to task count and diversity.",
        reference_explanation="Instruction tuning teaches language models to follow a wide variety of instructions — 'translate this,' 'summarize that,' 'classify this.' The more diverse the training tasks, the better the model becomes at following completely new instructions it's never encountered.",
        relevance_tags={"single_topic_retrieval": 0.15, "ambiguous_query_filtering": 0.1, "multi_concept_synthesis": 0.45},
    ),
    Paper(
        id="paper_042",
        title="Long Context Window Transformers: Extending to 100K+ Tokens",
        abstract="We present architectural innovations enabling transformers to process sequences exceeding 100,000 tokens. Our approach combines sliding window attention, sparse global tokens, and progressive training from short to long contexts, achieving strong performance on document-level tasks.",
        keywords=["long context", "extended context window", "sliding window attention", "sparse attention", "document-level", "100K tokens"],
        sub_topic="nlp_applications",
        reference_summary="Extends transformer context windows beyond 100K tokens via sliding window attention, sparse global tokens, and progressive length training, enabling effective document-level understanding.",
        reference_explanation="Standard transformers struggle with very long documents. This work extends their capacity to 100,000+ tokens through clever attention tricks and gradual training from short to long texts, enabling processing of entire books or large codebases in one pass.",
        relevance_tags={"single_topic_retrieval": 0.3, "ambiguous_query_filtering": 0.1, "multi_concept_synthesis": 0.2},
    ),
    Paper(
        id="paper_043",
        title="Hallucination in Large Language Models: Detection and Mitigation",
        abstract="We comprehensively study hallucination in LLMs — cases where models generate plausible-sounding but factually incorrect text. We propose a taxonomy of hallucination types, introduce detection benchmarks, and evaluate mitigation strategies including retrieval augmentation, self-consistency checks, and attribution training.",
        keywords=["hallucination", "LLM", "factuality", "detection", "mitigation", "attribution", "self-consistency"],
        sub_topic="nlp_applications",
        reference_summary="Taxonomizes LLM hallucination types, introduces detection benchmarks, and evaluates mitigations: retrieval augmentation, self-consistency checks, and attribution training for improved factual reliability.",
        reference_explanation="Language models sometimes make things up — confidently stating facts that are simply wrong. This paper categorizes these 'hallucinations,' builds tests to detect them, and evaluates fixes like checking multiple outputs for consistency and requiring the model to cite its sources.",
        relevance_tags={"single_topic_retrieval": 0.1, "ambiguous_query_filtering": 0.15, "multi_concept_synthesis": 0.3},
    ),
    Paper(
        id="paper_044",
        title="Semantic Text Similarity: From Word Embeddings to Sentence Transformers",
        abstract="We trace the evolution of semantic text similarity methods from word2vec through ELMo and BERT to specialized sentence transformers. We benchmark these approaches on STS tasks and provide practical guidance for choosing embedding methods based on task requirements and computational constraints.",
        keywords=["semantic similarity", "text embedding", "sentence transformers", "word2vec", "BERT", "benchmarking"],
        sub_topic="nlp_applications",
        reference_summary="Traces semantic similarity evolution from word2vec to sentence transformers, benchmarking STS performance and providing practical selection guidance based on task requirements and compute constraints.",
        reference_explanation="How do you measure if two sentences mean the same thing? This paper compares the journey from simple word vectors to sophisticated sentence-level models, showing which approach works best for different applications and budgets.",
        relevance_tags={"single_topic_retrieval": 0.15, "ambiguous_query_filtering": 0.1, "multi_concept_synthesis": 0.15},
    ),
    Paper(
        id="paper_045",
        title="Knowledge Distillation for Efficient Language Models",
        abstract="We study knowledge distillation methods for compressing large language models into smaller, deployable versions. We compare standard KD, attention transfer, progressive distillation, and task-specific calibration, finding that progressive approaches preserve the most capability per parameter.",
        keywords=["knowledge distillation", "model compression", "efficiency", "language model", "progressive distillation", "deployment"],
        sub_topic="nlp_applications",
        reference_summary="Studies knowledge distillation methods for LLM compression: standard KD, attention transfer, progressive distillation. Progressive approaches preserve the most capability per parameter for deployable models.",
        reference_explanation="Large language models are too expensive for many applications. Knowledge distillation 'teaches' a small, fast model to mimic a large, powerful one — like a student learning from a master teacher. The small model runs cheaply while retaining much of the original's capability.",
        relevance_tags={"single_topic_retrieval": 0.1, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.15},
    ),

    # =====================================================================
    # SUB-TOPIC 8: Miscellaneous / Edge Cases (papers 046-050)
    # =====================================================================
    Paper(
        id="paper_046",
        title="Federated Learning: Collaborative Training Without Sharing Data",
        abstract="We present federated learning, where multiple parties collaboratively train a shared model without exchanging raw data. We analyze communication efficiency, privacy guarantees, and convergence properties of federated optimization algorithms.",
        keywords=["federated learning", "privacy", "distributed training", "data sharing", "communication efficiency", "collaborative"],
        sub_topic="distributed_ml",
        reference_summary="Presents federated learning for collaborative model training without data sharing, analyzing communication efficiency, privacy preservation, and convergence of distributed optimization algorithms.",
        reference_explanation="Federated learning lets multiple organizations train AI together without sharing their private data. Each party trains locally and shares only model updates — enabling collaboration in sensitive domains like healthcare where data can't leave the building.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.1, "multi_concept_synthesis": 0.05},
    ),
    Paper(
        id="paper_047",
        title="Graph Neural Networks: Methods and Applications",
        abstract="We survey graph neural networks (GNNs), covering message passing, spectral methods, and spatial approaches. We review applications in molecular property prediction, social network analysis, and recommendation systems, highlighting the unique challenges of learning on non-Euclidean data.",
        keywords=["GNN", "graph neural network", "message passing", "spectral", "molecular", "social network", "recommendation"],
        sub_topic="graph_ml",
        reference_summary="Surveys GNN methods (message passing, spectral, spatial) and applications (molecular prediction, social networks, recommendations), highlighting challenges of non-Euclidean data modeling.",
        reference_explanation="Graph neural networks process data that has connections — like social networks, molecules, or recommendation networks. They learn by passing messages along connections, enabling AI to reason about relationships and structure rather than just isolated data points.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.05},
    ),
    Paper(
        id="paper_048",
        title="Data-Centric AI: Better Data Over Better Models",
        abstract="We advocate for a data-centric approach to AI, arguing that improving data quality, consistency, and coverage often yields better results than architectural innovations. We present systematic methods for data auditing, cleaning, augmentation, and active learning.",
        keywords=["data-centric AI", "data quality", "data cleaning", "augmentation", "active learning", "data auditing"],
        sub_topic="ml_practice",
        reference_summary="Advocates data-centric AI, demonstrating that systematic data quality improvement (cleaning, augmentation, active learning) often yields better results than architectural innovations.",
        reference_explanation="Instead of building fancier models, this paper argues we should focus on making our training data better. Fixing label errors, removing duplicates, and strategically collecting new data often improves performance more than any algorithmic innovation.",
        relevance_tags={"single_topic_retrieval": 0.05, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.05},
    ),
    Paper(
        id="paper_049",
        title="Mixture of Experts: Scaling Models Efficiently Through Conditional Computation",
        abstract="We present Mixture of Experts (MoE) as a framework for scaling model capacity without proportionally increasing compute. By routing each input to a subset of specialized expert networks, MoE models achieve the performance of a much larger dense model at a fraction of the cost.",
        keywords=["mixture of experts", "MoE", "conditional computation", "routing", "sparse models", "scaling", "efficiency"],
        sub_topic="ml_practice",
        reference_summary="MoE scales model capacity efficiently via conditional computation, routing inputs to specialized expert subsets. Achieves dense-model performance at fractional compute cost through learned routing strategies.",
        reference_explanation="Instead of using the entire model for every input, Mixture of Experts only activates the most relevant 'specialist' sub-networks. Like a hospital with different departments — each patient is routed to the right specialist, keeping costs down while maintaining quality.",
        relevance_tags={"single_topic_retrieval": 0.1, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.15},
    ),
    Paper(
        id="paper_050",
        title="Evaluating Large Language Models: A Comprehensive Benchmark Survey",
        abstract="We survey evaluation methodologies for large language models across reasoning, knowledge, code generation, instruction following, and safety dimensions. We compare existing benchmarks, discuss contamination issues, and propose best practices for meaningful LLM evaluation.",
        keywords=["LLM evaluation", "benchmarks", "reasoning", "safety", "contamination", "instruction following", "code generation"],
        sub_topic="ml_practice",
        reference_summary="Surveys LLM evaluation across reasoning, knowledge, code, instruction-following, and safety dimensions. Discusses benchmark contamination, compares methodologies, and proposes evaluation best practices.",
        reference_explanation="How do we measure if a language model is actually good? This paper surveys the major benchmarks used to evaluate LLMs — from logic puzzles to coding challenges to safety tests — and highlights problems like test data leaking into training data that can make evaluations misleading.",
        relevance_tags={"single_topic_retrieval": 0.1, "ambiguous_query_filtering": 0.05, "multi_concept_synthesis": 0.2},
    ),
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_paper_by_id(paper_id: str) -> Optional[Paper]:
    """Get a paper by its ID. Returns None if not found."""
    for paper in PAPER_CORPUS:
        if paper.id == paper_id:
            return paper
    return None


def search_papers(query_terms: str, top_k: int = 15) -> List[Paper]:
    """
    Simple BM25-like keyword search over paper corpus.
    Scores each paper by keyword overlap with query terms.
    Returns top_k papers sorted by relevance score (descending).
    """
    query_words = set(query_terms.lower().split())
    scored_papers = []

    for paper in PAPER_CORPUS:
        # Build searchable text from title, abstract, keywords
        searchable = (
            paper.title.lower() + " " +
            paper.abstract.lower() + " " +
            " ".join(kw.lower() for kw in paper.keywords)
        )
        searchable_words = set(searchable.split())

        # Score: weighted keyword overlap
        title_words = set(paper.title.lower().split())
        keyword_set = set(kw.lower() for kw in paper.keywords)

        # Title match weighs 3x, keyword match weighs 2x, abstract match weighs 1x
        title_overlap = len(query_words & title_words) * 3.0
        keyword_overlap = sum(
            2.0 for qw in query_words
            for kw in keyword_set
            if qw in kw or kw in qw
        )
        abstract_overlap = len(query_words & searchable_words) * 1.0

        score = title_overlap + keyword_overlap + abstract_overlap

        if score > 0:
            scored_papers.append((score, paper))

    # Sort by score descending, break ties by paper id
    scored_papers.sort(key=lambda x: (-x[0], x[1].id))

    return [paper for _, paper in scored_papers[:top_k]]
