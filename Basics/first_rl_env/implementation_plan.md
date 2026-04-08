# Research Paper Assistant — OpenEnv RL Environment

> **Goal**: Transform the existing `first_rl_env` echo environment into a production-grade RL environment where an AI agent learns to search, rank, summarize, and explain research papers.

---

## User Review Required

> [!IMPORTANT]
> This is a **major architectural overhaul** of every file in the project. The existing echo environment logic will be completely replaced. Review each section carefully before approving.

> [!WARNING]
> The simulated paper corpus is embedded in-memory (no external API calls during grading). This keeps the environment deterministic and fast (<20 min, 2 CPU, 8GB RAM) but means papers are pre-loaded fixtures rather than live ArXiv queries.

---

## 1. 🧩 Problem Formulation

### Objective
The agent operates as a **research assistant**. Given a natural-language research query, it must:
1. **Search** a corpus of papers and retrieve candidates
2. **Filter/Rank** papers by relevance
3. **Summarize** selected papers
4. **Explain** key findings in user-friendly language
5. **Finalize** a research brief

### Episode Lifecycle
```
reset(task_config) → initial observation (query + empty workspace)
  ↓
step(search(query_terms))     → retrieved paper list + reward
step(filter(paper_ids))       → filtered set + reward
step(summarize(paper_id))     → summary text + reward
step(explain(paper_id))       → explanation text + reward
step(finalize())              → final score + done=True
```

### Success Criteria
- **Score ∈ [0.0, 1.0]** computed by deterministic grader
- Score ≥ 0.5 = partial success, Score ≥ 0.8 = strong success
- Episode terminates on `finalize()` or after `MAX_STEPS` (default 15)

### Why This is a Real-World Task
- Researchers spend **30%+ of time** on literature review
- This mirrors actual workflows: query → retrieve → filter → synthesize → explain
- Directly applicable to building AI research assistants, automated lit reviews, and educational tools

---

## 2. 🗂️ Project Folder Structure

```
first_rl_env/
├── .env                           # API keys
├── __init__.py                    # Package exports
├── models.py                     # Pydantic Action/Observation models
├── client.py                     # EnvClient implementation
├── inference.py                  # Baseline LLM agent
├── openenv.yaml                  # OpenEnv manifest
├── pyproject.toml                # Dependencies
├── Dockerfile                    # Container definition
├── README.md                     # Documentation
├── requirements.txt              # Pip requirements
├── data/
│   ├── __init__.py
│   └── paper_corpus.py           # [NEW] Simulated paper database (50 papers)
├── tasks/
│   ├── __init__.py               # [NEW]
│   ├── easy.py                   # [NEW] Easy task definition
│   ├── medium.py                 # [NEW] Medium task definition
│   └── hard.py                   # [NEW] Hard task definition
├── graders/
│   ├── __init__.py               # [NEW]
│   └── grader.py                 # [NEW] Deterministic grading engine
└── server/
    ├── __init__.py
    ├── app.py                    # FastAPI application (modified)
    └── research_env.py           # [NEW] Core environment (replaces first_rl_env_environment.py)
```

### Purpose of Each New File

| File | Purpose |
|------|---------|
| `data/paper_corpus.py` | 50 pre-built research papers with titles, abstracts, keywords, ground-truth relevance scores, and reference summaries |
| `tasks/easy.py` | Simple single-topic query with 3 clearly relevant papers |
| `tasks/medium.py` | Ambiguous query requiring filtering from 15 candidates, summarization of top 5 |
| `tasks/hard.py` | Complex multi-concept query requiring synthesis across 8+ papers with cross-referencing |
| `graders/grader.py` | Deterministic grader computing relevance, correctness, completeness, explanation quality |
| `server/research_env.py` | Core `ResearchAssistantEnvironment` class with full step/reset/state logic |

---

## 3. 🔁 Environment Design

### 3.1 State (`state()`)

```python
class ResearchEnvState(State):
    episode_id: str
    step_count: int
    current_phase: str          # "search" | "filter" | "summarize" | "explain" | "finalize"
    query: str                  # The research query for this episode
    task_name: str              # "easy" | "medium" | "hard"
    retrieved_paper_ids: List[str]
    filtered_paper_ids: List[str]
    summaries: Dict[str, str]   # paper_id → agent-generated summary
    explanations: Dict[str, str] # paper_id → agent-generated explanation
    action_history: List[str]   # log of actions taken
    cumulative_reward: float
```

### 3.2 Observation (`ResearchObservation`)

```python
class ResearchObservation(Observation):
    query: str                              # The research query
    current_phase: str                      # Current expected phase
    available_actions: List[str]            # Valid action types for this phase
    retrieved_papers: List[PaperSummary]    # Papers found (title + abstract snippet)
    filtered_papers: List[PaperSummary]     # Papers after filtering
    last_summary: str                       # Most recent summary generated
    last_explanation: str                   # Most recent explanation generated
    step_count: int
    max_steps: int
    feedback: str                           # Environment feedback on last action
```

### 3.3 Action (`ResearchAction`)

```python
class ResearchAction(Action):
    action_type: str    # "search" | "filter" | "summarize" | "explain" | "finalize"
    query_terms: str = ""               # For search action
    paper_ids: List[str] = []           # For filter action
    paper_id: str = ""                  # For summarize/explain actions
    content: str = ""                   # Agent's summary/explanation text
```

### 3.4 Step Logic (Transition Function)

```python
def step(action: ResearchAction) -> ResearchObservation:
    # 1. Validate action type is allowed in current phase
    if action.action_type not in PHASE_TRANSITIONS[current_phase]:
        return observation(reward=-0.1, feedback="Invalid action for phase")

    # 2. Execute action
    match action.action_type:
        case "search":
            # Search corpus using BM25-like keyword matching
            # Return top-K papers sorted by relevance
            # Reward: proportional to relevance of retrieved papers
            retrieved = search_corpus(action.query_terms)
            reward = compute_retrieval_reward(retrieved, ground_truth)
            next_phase = "filter"

        case "filter":
            # Agent selects subset of retrieved papers
            # Reward: precision/recall against ground truth relevant set
            filtered = validate_paper_ids(action.paper_ids)
            reward = compute_filter_reward(filtered, ground_truth)
            next_phase = "summarize"

        case "summarize":
            # Agent provides summary of a specific paper
            # Reward: ROUGE-L + keyword overlap with reference summary
            reward = compute_summary_reward(action.content, reference)
            next_phase = "summarize" or "explain"  # can summarize multiple

        case "explain":
            # Agent explains paper in accessible language
            # Reward: readability + accuracy + completeness
            reward = compute_explanation_reward(action.content, reference)
            next_phase = "explain" or "finalize"

        case "finalize":
            # Terminal action — compute final composite score
            final_score = grader.grade(state)
            return observation(reward=final_score, done=True)

    # 3. Penalize redundant actions
    if action in action_history:
        reward *= 0.5  # diminishing returns

    # 4. Check step limit
    if step_count >= MAX_STEPS:
        done = True
        reward += grader.grade(state) * 0.5  # partial credit

    return observation(reward=reward, done=done)
```

### Phase Transition Rules

```
search → filter → summarize ↔ explain → finalize
                      ↑___________|
```

| From | Allowed Next |
|------|-------------|
| `search` | `filter`, `search` (refine) |
| `filter` | `summarize`, `filter` (re-filter) |
| `summarize` | `summarize` (another paper), `explain`, `finalize` |
| `explain` | `explain` (another paper), `summarize`, `finalize` |
| `finalize` | *(terminal)* |

---

## 4. 🎯 Task Design

### 4.1 Easy Task: `"single_topic_retrieval"`

**Query**: `"What is the transformer architecture in NLP?"`

**Corpus**: 10 papers, 3 clearly relevant (attention mechanism, BERT, GPT)

**Expected Behavior**:
1. Search with keywords like `"transformer attention NLP"`
2. Filter to 3 relevant papers
3. Summarize 1–2 papers
4. Explain 1 paper
5. Finalize

**Ground Truth**:
- Relevant papers: `["paper_001", "paper_003", "paper_007"]`
- Reference summary keywords: `["self-attention", "encoder-decoder", "positional encoding"]`
- Reference explanation must mention: `["parallel processing", "sequence modeling"]`

**Edge Cases**:
- Agent searches with overly broad terms → gets noisy results → lower retrieval reward
- Agent skips summarization → finalize score penalized for incompleteness

---

### 4.2 Medium Task: `"ambiguous_query_filtering"`

**Query**: `"How do neural networks handle uncertainty?"`

**Corpus**: 25 papers across Bayesian NNs, dropout regularization, ensemble methods, unrelated CV papers

**Expected Behavior**:
1. Search with nuanced terms — `"neural network uncertainty quantification"`
2. Filter from ~12 candidates to 5 most relevant
3. Summarize 3 papers covering different approaches
4. Explain the key trade-offs between methods
5. Finalize

**Ground Truth**:
- Relevant papers: 5 papers across 3 sub-topics (Bayesian, dropout-as-inference, ensembles)
- Reference summaries must cover at least 2 sub-topics
- Explanation should compare/contrast approaches

**Edge Cases**:
- Agent retrieves only dropout papers → misses Bayesian → lower completeness score
- Agent summarizes redundant papers from same sub-topic → diminishing reward

---

### 4.3 Hard Task: `"multi_concept_synthesis"`

**Query**: `"What are the recent advances combining reinforcement learning with large language models for decision-making?"`

**Corpus**: 50 papers — RL fundamentals, LLM fine-tuning, RLHF, constitutional AI, tool use, planning, unrelated papers

**Expected Behavior**:
1. Search with compound terms — `"reinforcement learning language model decision making"`
2. Filter from ~20 candidates to 8 truly relevant
3. Summarize 5+ papers across sub-themes (RLHF, planning, tool use)
4. Explain how these advances interconnect — synthesis, not just listing
5. Finalize

**Ground Truth**:
- Relevant papers: 8 papers across 4 sub-themes
- Summaries must capture key contributions of each paper
- Explanation must show **cross-paper connections** (e.g., "RLHF enables X, which combined with tool-use Y leads to Z")

**Edge Cases**:
- Agent treats this as simple retrieval → misses synthesis → low explanation score
- Agent hallucinates paper content → detected via keyword mismatch → penalty
- Agent loops on summarize without progressing to explain → step limit reached

---

## 5. 🧪 Grader Design

### Grading Formula

For each task, the final score is:

```
score = w_r * relevance + w_c * correctness + w_comp * completeness + w_e * explanation_quality
```

| Task | w_r (Relevance) | w_c (Correctness) | w_comp (Completeness) | w_e (Explanation) |
|------|:---:|:---:|:---:|:---:|
| Easy | 0.30 | 0.30 | 0.20 | 0.20 |
| Medium | 0.25 | 0.25 | 0.25 | 0.25 |
| Hard | 0.20 | 0.20 | 0.25 | 0.35 |

### Individual Metric Computation

#### `relevance` ∈ [0, 1]
```python
# Precision-Recall F1 of filtered papers vs ground truth
precision = |filtered ∩ ground_truth| / |filtered|
recall = |filtered ∩ ground_truth| / |ground_truth|
relevance = 2 * precision * recall / (precision + recall + ε)
```

#### `correctness` ∈ [0, 1]
```python
# Keyword overlap between agent summaries and reference summaries
for each summarized paper:
    keywords_hit = |agent_keywords ∩ reference_keywords| / |reference_keywords|
correctness = mean(keywords_hit across all summarized papers)
```

#### `completeness` ∈ [0, 1]
```python
# How many required sub-topics were covered?
sub_topics_covered = set()
for summary in agent_summaries:
    for topic in required_sub_topics:
        if topic_keywords overlap with summary:
            sub_topics_covered.add(topic)
completeness = |sub_topics_covered| / |required_sub_topics|
```

#### `explanation_quality` ∈ [0, 1]
```python
# Composite of: readability + accuracy + synthesis
readability = 1.0 if avg_sentence_length < 25 words and no jargon overload else scaled
accuracy = keyword_overlap(agent_explanation, reference_explanation)
synthesis = 1.0 if cross_references_detected(explanation) else 0.5  # Hard task bonus
explanation_quality = 0.3 * readability + 0.4 * accuracy + 0.3 * synthesis
```

---

## 6. 🎁 Reward Function Design

### Dense Reward per Step

| Action | Reward Formula | Range |
|--------|---------------|-------|
| `search` | `R_search = 0.1 * (relevant_in_top10 / total_relevant)` | [0, 0.1] |
| `filter` | `R_filter = 0.15 * F1(filtered, ground_truth)` | [0, 0.15] |
| `summarize` | `R_summary = 0.2 * keyword_overlap(agent, ref)` | [0, 0.2] |
| `explain` | `R_explain = 0.2 * explanation_quality_score` | [0, 0.2] |
| `finalize` | `R_final = grader.grade(state)` — full composite | [0, 1.0] |

### Penalties

| Violation | Penalty |
|-----------|---------|
| Invalid action for current phase | `-0.1` |
| Repeated identical action | `reward *= 0.5` (halved) |
| Hallucinated paper_id (not in corpus) | `-0.15` |
| Empty content in summarize/explain | `-0.1` |
| Step limit exceeded without finalize | `reward = 0.5 * grader.grade(state)` |

### Mathematical Intuition

```
R_total = Σ(R_step_i) for i in 1..N
score = R_final / max_possible_score   # normalized to [0, 1]

Where max_possible_score ≈ 0.1 + 0.15 + K*0.2 + K*0.2 + 1.0
  (K = number of papers summarized/explained)
```

The reward is **dense** — every step yields meaningful signal. The agent is incentivized to:
1. Search well (retrieval covers ground truth)
2. Filter precisely (high F1)
3. Summarize accurately (keyword coverage)
4. Explain clearly (readability + synthesis)
5. Finalize at the right time (not too early, not at step limit)

---

## 7. 🤖 Baseline Agent (`inference.py`)

### Design

```python
# Environment variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("RESEARCH_TASK", "single_topic_retrieval")
BENCHMARK = os.getenv("RESEARCH_BENCHMARK", "research_assistant_env")
MAX_STEPS = 15

SYSTEM_PROMPT = """
You are a research assistant agent interacting with a paper search environment.
At each step, you must choose ONE action based on the current phase.

Phases and actions:
- search phase: emit action_type="search" with query_terms
- filter phase: emit action_type="filter" with paper_ids (list of IDs to keep)
- summarize phase: emit action_type="summarize" with paper_id and content (your summary)
- explain phase: emit action_type="explain" with paper_id and content (your explanation)
- finalize phase: emit action_type="finalize"

Respond ONLY with valid JSON matching the action schema.
"""
```

### Step-by-Step Interaction

```
[START] task=single_topic_retrieval env=research_assistant_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=search:transformer+attention+NLP reward=0.08 done=false error=null
[STEP] step=2 action=filter:paper_001,paper_003,paper_007 reward=0.15 done=false error=null
[STEP] step=3 action=summarize:paper_001 reward=0.16 done=false error=null
[STEP] step=4 action=summarize:paper_003 reward=0.14 done=false error=null
[STEP] step=5 action=explain:paper_001 reward=0.18 done=false error=null
[STEP] step=6 action=finalize reward=0.82 done=true error=null
[END] success=true steps=6 score=0.820 rewards=0.08,0.15,0.16,0.14,0.18,0.82
```

---

## 8. 🐳 Docker + Deployment

### Dockerfile Changes
- Base image remains `ghcr.io/meta-pytorch/openenv-base:latest`
- Add `nltk` data download for tokenization (used in grading)
- No external API calls during grading — everything is in-memory
- Runtime: **<5 min per episode**, well within 20 min limit

### Requirements Additions
```
nltk>=3.8
pydantic>=2.0
huggingface-hub>=1.9.0
openenv-core[core]>=0.2.2
```

### HF Spaces Deployment
- Use `openenv push` as documented
- Set `app_port: 8000` in README frontmatter
- Environment variables configured in Space settings: `HF_TOKEN`, `MODEL_NAME`

---

## 9. 📄 `openenv.yaml` Design

```yaml
spec_version: 1
name: research_assistant_env
type: space
runtime: fastapi
app: server.app:app
port: 8000

description: >
  RL environment where an AI agent learns to search, filter, summarize,
  and explain research papers given a natural language query.

tasks:
  - name: single_topic_retrieval
    difficulty: easy
    description: Simple single-topic paper search and summary

  - name: ambiguous_query_filtering
    difficulty: medium
    description: Filtering relevant papers from an ambiguous query

  - name: multi_concept_synthesis
    difficulty: hard
    description: Multi-paper synthesis across interconnected research themes

action_schema:
  type: object
  properties:
    action_type:
      type: string
      enum: [search, filter, summarize, explain, finalize]
    query_terms:
      type: string
    paper_ids:
      type: array
      items: { type: string }
    paper_id:
      type: string
    content:
      type: string

observation_schema:
  type: object
  properties:
    query: { type: string }
    current_phase: { type: string }
    available_actions: { type: array, items: { type: string } }
    retrieved_papers: { type: array }
    filtered_papers: { type: array }
    last_summary: { type: string }
    last_explanation: { type: string }
    step_count: { type: integer }
    max_steps: { type: integer }
    feedback: { type: string }
```

---

## 10. 📘 README Structure

The README will include:
1. **HF Spaces frontmatter** (title, emoji, sdk, port, tags)
2. **Overview** — what the environment does
3. **Motivation** — real-world research assistant use case
4. **Environment Design** — state/observation/action lifecycle diagram
5. **Action & Observation Space** — schema tables
6. **Tasks** — easy/medium/hard with descriptions
7. **Setup Instructions** — virtualenv, uv sync, Docker build
8. **Run Instructions** — inference script, environment variables
9. **Baseline Results** — expected scores per task
10. **Deployment** — HF Spaces push instructions

---

## 11. 💡 Innovation & Edge

### Why This Environment is Novel
- **Multi-phase sequential decision making** — not a single-shot task but a pipeline the agent must orchestrate
- **Grounded in real academic workflow** — mirrors how researchers actually work
- **Dense reward signal at every step** — not just final binary success, enables actual RL training
- **Deterministic grading** — reproducible evaluation, no randomness in scoring

### Scoring High on Hackathon Criteria

| Criterion (Weight) | Why We Score High |
|--------------------|--------------------|
| **Real-world utility (30%)** | Direct application to research assistants, lit review automation, educational tools. Researchers spend 30%+ time on literature review. |
| **Task quality (25%)** | 3 well-differentiated tasks with ground truth, edge cases, and deterministic grading. Easy→Medium→Hard progression tests retrieval→filtering→synthesis skills. |
| **Environment design (20%)** | Clean phase-based FSM, typed Pydantic models, dense rewards, penalty system, proper OpenEnv compliance with step/reset/state. |

---

## Proposed Changes

### Root Level

#### [MODIFY] [models.py](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/models.py)
Replace `FirstRlAction`/`FirstRlObservation` with `ResearchAction`/`ResearchObservation` + `PaperSummary` model.

#### [MODIFY] [client.py](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/client.py)
Replace `FirstRlEnv` with `ResearchAssistantEnv` client, update `_step_payload`, `_parse_result`, `_parse_state`.

#### [MODIFY] [inference.py](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/inference.py)
Complete rewrite: multi-phase LLM agent that emits structured JSON actions per phase.

#### [MODIFY] [__init__.py](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/__init__.py)
Update exports.

#### [MODIFY] [openenv.yaml](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/openenv.yaml)
Full replacement with new schema.

#### [MODIFY] [pyproject.toml](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/pyproject.toml)
Update name, description, add `nltk` dependency.

#### [MODIFY] [Dockerfile](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/Dockerfile)
Add NLTK data download step.

#### [MODIFY] [README.md](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/README.md)
Complete rewrite with new environment documentation.

---

### Data Layer

#### [NEW] [data/__init__.py](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/data/__init__.py)
Package init.

#### [NEW] [data/paper_corpus.py](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/data/paper_corpus.py)
50 simulated research papers with: id, title, abstract, keywords, sub_topic, reference_summary, reference_explanation.

---

### Tasks Layer

#### [NEW] [tasks/__init__.py](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/tasks/__init__.py)
Task registry and loader.

#### [NEW] [tasks/easy.py](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/tasks/easy.py)
`single_topic_retrieval` task config.

#### [NEW] [tasks/medium.py](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/tasks/medium.py)
`ambiguous_query_filtering` task config.

#### [NEW] [tasks/hard.py](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/tasks/hard.py)
`multi_concept_synthesis` task config.

---

### Graders Layer

#### [NEW] [graders/__init__.py](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/graders/__init__.py)
Package init.

#### [NEW] [graders/grader.py](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/graders/grader.py)
`ResearchGrader` class with deterministic scoring: relevance (F1), correctness (keyword overlap), completeness (sub-topic coverage), explanation quality (readability + accuracy + synthesis).

---

### Server Layer

#### [NEW] [server/research_env.py](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/server/research_env.py)
`ResearchAssistantEnvironment(Environment)` — the core environment class with reset(), step(), state(). Replaces `first_rl_env_environment.py`.

#### [MODIFY] [server/app.py](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/server/app.py)
Update imports to use `ResearchAssistantEnvironment`, `ResearchAction`, `ResearchObservation`.

#### [MODIFY] [server/__init__.py](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/server/__init__.py)
Update exports.

#### [DELETE] [server/first_rl_env_environment.py](file:///home/sayandip-saha/Desktop/CODING/RL_Environment_Design/Basics/first_rl_env/server/first_rl_env_environment.py)
Replaced by `server/research_env.py`.

---

## Open Questions

> [!IMPORTANT]
> **1. Paper Corpus Source**: Should the 50 papers be based on real ArXiv paper metadata (title/abstract from public datasets), or fully synthetic? Real metadata would increase authenticity but requires embedding ~50 paper abstracts in the code.

> [!IMPORTANT]
> **2. Task Selection**: The current `inference.py` reads `FIRST_RL_TASK` env var. Should the new env support running all 3 tasks in a single inference run, or one task per run (matching the current pattern)?

> [!WARNING]
> **3. NLTK Dependency**: The grader uses NLTK for tokenization (sentence splitting, word tokenization). This adds ~15MB to the Docker image. An alternative is simple regex-based splitting. Which do you prefer?

---

## Verification Plan

### Automated Tests
1. `python -c "from server.research_env import ResearchAssistantEnvironment; env = ResearchAssistantEnvironment(); obs = env.reset(); print(obs)"` — verify reset works
2. Run full episode manually with each task: `RESEARCH_TASK=single_topic_retrieval python inference.py`
3. Verify grader scoring is deterministic: run same actions twice, assert identical scores
4. `docker build -t research-env:latest .` — verify Docker builds
5. Verify STDOUT format matches `[START]/[STEP]/[END]` spec

### Manual Verification
- Review paper corpus for reasonable quality
- Verify grader scores make intuitive sense for good vs bad agent behavior
- Test that invalid actions are properly penalized
