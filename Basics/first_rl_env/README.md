---
title: Research Paper Assistant Environment
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - research-assistant
  - paper-search
---

# 📚 Research Paper Assistant — OpenEnv RL Environment

An RL environment where an AI agent learns to **search**, **filter**, **summarize**, and **explain** research papers — simulating the real-world workflow of an academic research assistant.

## 🎯 Overview

This environment challenges an AI agent with a multi-phase research workflow:

1. **Search** — Query a corpus of 50 ML/AI papers using keyword search
2. **Filter** — Select the most relevant papers from search results
3. **Summarize** — Write technical summaries capturing key contributions
4. **Explain** — Produce accessible, user-friendly explanations
5. **Finalize** — Signal completion and receive a composite grading score

The agent receives **dense rewards** at every step, enabling practical RL training. All grading is **deterministic** — no randomness, no external API calls during evaluation.

## 💡 Motivation

Researchers spend **30%+ of their time** on literature review. This environment directly addresses that pain point by training agents to:

- Navigate large paper corpora efficiently
- Distinguish relevant from irrelevant work
- Synthesize findings across multiple papers
- Communicate complex ideas accessibly

Unlike toy environments, this simulates a **production-grade research assistant** workflow with realistic challenges: ambiguous queries, noisy retrieval, multi-topic synthesis.

## 🔁 Environment Design

### Phase-Based Finite State Machine

```
search → filter → summarize ↔ explain → finalize
                      ↑___________|
```

| From | Allowed Next Actions |
|------|---------------------|
| `search` | `search` (refine), `filter` |
| `filter` | `filter` (re-filter), `summarize` |
| `summarize` | `summarize` (more papers), `explain`, `finalize` |
| `explain` | `explain` (more papers), `summarize`, `finalize` |

### Episode Lifecycle

```python
obs = env.reset()           # → query + empty workspace
obs = env.step(search(...)) # → retrieved papers + reward
obs = env.step(filter(...)) # → filtered set + reward
obs = env.step(summarize()) # → summary feedback + reward
obs = env.step(explain())   # → explanation feedback + reward
obs = env.step(finalize())  # → final score (done=True)
```

## 🎮 Action Space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `str` | One of: `search`, `filter`, `summarize`, `explain`, `finalize` |
| `query_terms` | `str` | Search keywords (for `search`) |
| `paper_ids` | `List[str]` | Paper IDs to keep (for `filter`) |
| `paper_id` | `str` | Target paper (for `summarize`/`explain`) |
| `content` | `str` | Agent-generated text (for `summarize`/`explain`) |

## 👁️ Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `query` | `str` | Research question for this episode |
| `current_phase` | `str` | Current FSM phase |
| `available_actions` | `List[str]` | Valid action types |
| `retrieved_papers` | `List[PaperInfo]` | Search results (id, title, abstract snippet, keywords) |
| `filtered_papers` | `List[PaperInfo]` | Agent-selected papers |
| `summaries_so_far` | `Dict[str, str]` | paper_id → summary text |
| `explanations_so_far` | `Dict[str, str]` | paper_id → explanation text |
| `last_action_feedback` | `str` | Environment feedback |
| `step_count` | `int` | Current step |
| `max_steps` | `int` | Step limit (default: 15) |

## 📋 Tasks

### Easy: `single_topic_retrieval`
- **Query**: "What is the transformer architecture in NLP?"
- **Corpus**: 10 candidates, 3 clearly relevant
- **Goal**: Basic search → filter → summarize → explain

### Medium: `ambiguous_query_filtering`
- **Query**: "How do neural networks handle uncertainty?"
- **Corpus**: 25 candidates across Bayesian/dropout/ensemble sub-topics
- **Goal**: Distinguish approaches, cover multiple sub-topics

### Hard: `multi_concept_synthesis`
- **Query**: "What are the recent advances combining RL with LLMs for decision-making?"
- **Corpus**: 50 papers across 4 sub-themes
- **Goal**: Synthesize cross-paper connections, explain interconnected advances

## 🧪 Grading

Each task is scored on 4 metrics (weighted by difficulty):

| Metric | Easy | Medium | Hard |
|--------|:----:|:------:|:----:|
| **Relevance** (F1 of filtered vs ground truth) | 0.30 | 0.25 | 0.20 |
| **Correctness** (keyword overlap in summaries) | 0.30 | 0.25 | 0.20 |
| **Completeness** (sub-topic + count coverage) | 0.20 | 0.25 | 0.25 |
| **Explanation Quality** (readability + accuracy + synthesis) | 0.20 | 0.25 | 0.35 |

Final score ∈ [0.0, 1.0]. All grading is **deterministic**.

## 🎁 Reward Function

Dense rewards per step:
- `search`: 0–0.10 (proportional to ground-truth recall)
- `filter`: 0–0.15 (F1 against ground truth)
- `summarize`: 0–0.20 (keyword overlap with reference)
- `explain`: 0–0.20 (readability + accuracy)
- `finalize`: 0–1.0 (full composite grader score)

Penalties:
- Invalid action for phase: −0.10
- Repeated identical action: reward halved
- Hallucinated paper ID: −0.15
- Empty content: −0.10

## 🚀 Setup Instructions

### Prerequisites
- Python ≥ 3.10
- `uv` package manager (recommended) or `pip`

### Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -e .
```

### Environment Variables

```bash
export HF_TOKEN="your-huggingface-token"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"  # or any HF model
export RESEARCH_TASK="single_topic_retrieval"     # easy | medium | hard
```

## ▶️ Run Instructions

### Run Inference (Baseline Agent)

```bash
# Easy task
RESEARCH_TASK=single_topic_retrieval python inference.py

# Medium task
RESEARCH_TASK=ambiguous_query_filtering python inference.py

# Hard task
RESEARCH_TASK=multi_concept_synthesis python inference.py
```

### Run Server Locally

```bash
# Development mode
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Direct execution
python -m server.app
```

### Docker

```bash
# Build
docker build -t research-assistant-env:latest .

# Run
docker run -p 8000:8000 -e RESEARCH_TASK=single_topic_retrieval research-assistant-env:latest
```

### Deploy to Hugging Face Spaces

```bash
openenv push
# Or with options:
openenv push --repo-id your-username/research-assistant-env --private
```

## 📊 Baseline Results

Expected scores with `Qwen/Qwen2.5-72B-Instruct`:

| Task | Expected Score | Steps |
|------|:-----------:|:-----:|
| Easy | 0.65–0.85 | 5–7 |
| Medium | 0.45–0.70 | 8–12 |
| Hard | 0.30–0.55 | 10–15 |

## 📁 Project Structure

```
research_assistant_env/
├── __init__.py                  # Package exports
├── models.py                    # Pydantic Action/Observation models
├── client.py                    # WebSocket/HTTP client
├── inference.py                 # Baseline LLM agent
├── openenv.yaml                 # OpenEnv manifest
├── pyproject.toml               # Project metadata
├── Dockerfile                   # Container definition
├── README.md                    # This file
├── data/
│   ├── __init__.py
│   └── paper_corpus.py          # 50 simulated research papers
├── tasks/
│   ├── __init__.py              # Task registry
│   ├── easy.py                  # single_topic_retrieval
│   ├── medium.py                # ambiguous_query_filtering
│   └── hard.py                  # multi_concept_synthesis
├── graders/
│   ├── __init__.py
│   └── grader.py                # Deterministic scoring engine
└── server/
    ├── __init__.py
    ├── research_env.py           # Core environment logic
    └── app.py                    # FastAPI server
```
