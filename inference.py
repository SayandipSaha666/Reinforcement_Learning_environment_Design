"""
Inference Script for Research Paper Assistant Environment
============================================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    OPENAI_API_KEY Your OpenAI API key.

- Defaults are set only for MODEL_NAME
    (and should reflect your active inference setup):
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line at episode end, always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw error message string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]

  Example:
    [START] task=single_topic_retrieval env=research_assistant_env model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action=search:transformer+attention+NLP reward=0.08 done=false error=null
    [STEP] step=2 action=filter:paper_001,paper_003,paper_007 reward=0.15 done=false error=null
    [STEP] step=3 action=summarize:paper_001 reward=0.16 done=false error=null
    [STEP] step=4 action=explain:paper_001 reward=0.18 done=false error=null
    [STEP] step=5 action=finalize reward=0.82 done=true error=null
    [END] success=true steps=5 score=0.820 rewards=0.08,0.15,0.16,0.18,0.82
"""

import json
import os
import textwrap
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from models import ResearchAction, ResearchObservation
from server.research_env import ResearchAssistantEnvironment

BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")

if not API_KEY:
    raise RuntimeError("Missing API key: set API_KEY or OPENAI_API_KEY in the environment.")
if not BASE_URL:
    raise RuntimeError("Missing API base URL: set API_BASE_URL in the environment.")

MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"
TASK_NAME = os.getenv("RESEARCH_TASK", "single_topic_retrieval")
BENCHMARK = os.getenv("RESEARCH_BENCHMARK", "research_assistant_env")
MAX_STEPS = 15
TEMPERATURE = 0.7
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.3  # normalized score in [0, 1]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a research paper assistant agent interacting with an RL environment.
    Your goal is to search, filter, summarize, and explain research papers based on a query.

    At each step, you receive an observation containing:
    - query: the research question you're helping with
    - current_phase: what type of action is expected next
    - available_actions: valid action types you can choose
    - retrieved_papers: papers found by search (with paper_id, title, keywords)
    - filtered_papers: papers you've selected as relevant
    - summaries_so_far: summaries you've written
    - explanations_so_far: explanations you've written
    - last_action_feedback: feedback from the environment

    You must respond with a JSON object matching one of these schemas:

    For "search" phase:
    {"action_type": "search", "query_terms": "your search keywords"}

    For "filter" phase:
    {"action_type": "filter", "paper_ids": ["paper_001", "paper_003"]}

    For "summarize" phase:
    {"action_type": "summarize", "paper_id": "paper_001", "content": "Your technical summary of the paper..."}

    For "explain" phase:
    {"action_type": "explain", "paper_id": "paper_001", "content": "Your user-friendly explanation..."}

    For "finalize" phase:
    {"action_type": "finalize"}

    RULES:
    - Respond with ONLY the JSON object, no markdown formatting, no extra text.
    - Search with specific, relevant keywords from the query.
    - Filter to papers most relevant to the query.
    - Summaries should capture key technical contributions and methods.
    - Explanations should be accessible to non-experts, using analogies.
    - Finalize when you've summarized and explained enough papers.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(observation: ResearchObservation, step: int) -> str:
    """Build context-rich prompt from current observation."""
    parts = [
        f"Step: {step}/{observation.max_steps}",
        f"Query: {observation.query}",
        f"Current phase: {observation.current_phase}",
        f"Available actions: {observation.available_actions}",
        f"Feedback: {observation.last_action_feedback}",
    ]

    if observation.retrieved_papers:
        papers_list = []
        for p in observation.retrieved_papers:
            papers_list.append(
                f"  - {p.paper_id}: {p.title} [keywords: {', '.join(p.keywords[:5])}]"
            )
        parts.append(f"Retrieved papers ({len(observation.retrieved_papers)}):")
        parts.extend(papers_list)

    if observation.filtered_papers:
        parts.append(f"Filtered papers: {[p.paper_id for p in observation.filtered_papers]}")

    if observation.summaries_so_far:
        parts.append(f"Papers summarized: {list(observation.summaries_so_far.keys())}")

    if observation.explanations_so_far:
        parts.append(f"Papers explained: {list(observation.explanations_so_far.keys())}")

    parts.append("\nRespond with the appropriate JSON action.")

    return "\n".join(parts)


def parse_model_response(response_text: str) -> dict:
    """Parse LLM response into action dict, handling common formatting issues."""
    text = response_text.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from text
    import re
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return a finalize action to end gracefully
    print(f"[DEBUG] Could not parse model response: {text[:200]}", flush=True)
    return {"action_type": "finalize"}


def get_model_action(
    client: OpenAI,
    observation: ResearchObservation,
    step: int,
    history: List[str],
) -> dict:
    """Get next action from LLM based on current observation."""
    user_prompt = build_user_prompt(observation, step)

    try:
        messages: list[ChatCompletionMessageParam]  = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_model_response(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # On failure, try to finalize
        return {"action_type": "finalize"}


def make_action(action_dict: dict) -> ResearchAction:
    """Convert action dict to ResearchAction, handling missing fields."""
    return ResearchAction(
        action_type=action_dict.get("action_type", "finalize"),
        query_terms=action_dict.get("query_terms", ""),
        paper_ids=action_dict.get("paper_ids", []),
        paper_id=action_dict.get("paper_id", ""),
        content=action_dict.get("content", ""),
    )


def format_action_str(action: ResearchAction) -> str:
    """Format action for logging — compact single-line representation."""
    if action.action_type == "search":
        return f"search:{action.query_terms.replace(' ', '+')}"
    elif action.action_type == "filter":
        return f"filter:{','.join(action.paper_ids)}"
    elif action.action_type == "summarize":
        snippet = action.content[:40].replace(" ", "_") if action.content else "empty"
        return f"summarize:{action.paper_id}:{snippet}"
    elif action.action_type == "explain":
        snippet = action.content[:40].replace(" ", "_") if action.content else "empty"
        return f"explain:{action.paper_id}:{snippet}"
    elif action.action_type == "finalize":
        return "finalize"
    return f"unknown:{action.action_type}"


def main() -> None:
    client = OpenAI(api_key=API_KEY,base_url=BASE_URL)

    # Set task via env var so environment picks it up
    os.environ["RESEARCH_TASK"] = TASK_NAME

    env = ResearchAssistantEnvironment()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset()

        for step in range(1, MAX_STEPS + 1):
            if observation.done:
                break

            # Get action from LLM
            action_dict = get_model_action(client, observation, step, history)
            action = make_action(action_dict)

            # Step environment
            observation = env.step(action)

            reward = observation.reward or 0.0
            done = observation.done
            error = None

            rewards.append(reward)
            steps_taken = step

            action_str = format_action_str(action)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        # Compute final score
        # The last reward on finalize IS the composite grader score
        if rewards and observation.done:
            score = rewards[-1]  # finalize reward = grader composite score
        else:
            # If we didn't finalize, estimate score from cumulative
            score = max(0.0, min(1.0, sum(rewards)))

        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Error during inference: {e}", flush=True)
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()