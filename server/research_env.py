"""
Research Paper Assistant Environment Implementation
=====================================================

A multi-phase RL environment where an AI agent learns to search, filter,
summarize, and explain research papers. Phases follow a finite-state
machine: search → filter → summarize ↔ explain → finalize.

Compliant with OpenEnv spec: step(action) → observation, reset() → observation,
state → internal state.
"""

import os
from typing import Dict, List, Optional, Set
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from data.paper_corpus import get_paper_by_id, search_papers
    from graders.grader import ResearchGrader
    from models import PaperInfo, ResearchAction, ResearchObservation
    from tasks import get_task
except ImportError:
    from ..data.paper_corpus import get_paper_by_id, search_papers
    from ..graders.grader import ResearchGrader
    from ..models import PaperInfo, ResearchAction, ResearchObservation
    from ..tasks import get_task


# Phase transition rules — maps current_phase to list of valid next action types
PHASE_TRANSITIONS: Dict[str, List[str]] = {
    "search": ["search", "filter"],
    "filter": ["filter", "summarize"],
    "summarize": ["summarize", "explain", "finalize"],
    "explain": ["explain", "summarize", "finalize"],
    "finalize": [],  # terminal
}

DEFAULT_TASK = os.getenv("RESEARCH_TASK", "single_topic_retrieval")
MAX_STEPS = int(os.getenv("RESEARCH_MAX_STEPS", "15"))


class ResearchAssistantEnvironment(Environment):
    """
    Multi-phase RL environment for research paper search, filter, summarize,
    and explain.

    Episode lifecycle:
        reset() → observation (query + empty workspace)
        step(search)   → retrieved papers + reward
        step(filter)   → filtered set + reward
        step(summarize)→ summary feedback + reward
        step(explain)  → explanation feedback + reward
        step(finalize) → final score + done=True

    Example:
        >>> env = ResearchAssistantEnvironment()
        >>> obs = env.reset()
        >>> print(obs.query)  # "What is the transformer architecture in NLP?"
        >>> obs = env.step(ResearchAction(action_type="search", query_terms="transformer attention NLP"))
        >>> print(len(obs.retrieved_papers))  # e.g. 10
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the research assistant environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_name = DEFAULT_TASK
        self._task = get_task(self._task_name)
        self._grader = ResearchGrader(self._task)
        self._custom_query = os.getenv("RESEARCH_QUERY", "").strip()

        # Episode state
        self._current_phase: str = "search"
        self._retrieved_paper_ids: List[str] = []
        self._filtered_paper_ids: List[str] = []
        self._summaries: Dict[str, str] = {}
        self._explanations: Dict[str, str] = {}
        self._action_history: List[str] = []
        self._cumulative_reward: float = 0.0
        self._max_steps: int = MAX_STEPS

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> ResearchObservation:
        """
        Reset the environment for a new episode.

        Args:
            seed: optional RNG seed from the OpenEnv runtime.
            episode_id: optional episode identifier.
            **kwargs: runtime-provided options such as task_name or query.

        Returns:
            ResearchObservation with the research query and initial state.
        """
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)

        # Allow OpenEnv runtime to specify task_name or task
        task_name = kwargs.get("task_name") or kwargs.get("task") or os.getenv("RESEARCH_TASK", DEFAULT_TASK)
        self._task_name = task_name
        self._task = get_task(task_name)
        self._grader = ResearchGrader(self._task)

        # Allow OpenEnv runtime to supply a custom query for this episode
        self._custom_query = kwargs.get("query", os.getenv("RESEARCH_QUERY", "")).strip()

        # Reset episode state
        self._current_phase = "search"
        self._retrieved_paper_ids = []
        self._filtered_paper_ids = []
        self._summaries = {}
        self._explanations = {}
        self._action_history = []
        self._cumulative_reward = 0.0

        return ResearchObservation(
            query=self._custom_query if self._custom_query else self._task.query,
            current_phase="search",
            available_actions=["search"],
            retrieved_papers=[],
            filtered_papers=[],
            summaries_so_far={},
            explanations_so_far={},
            last_action_feedback=f"Episode started. Task: {self._task.name} ({self._task.difficulty}). Please search for relevant papers.",
            step_count=0,
            max_steps=self._max_steps,
            done=False,
            reward=0.0,
        )

    def step(self, action: ResearchAction) -> ResearchObservation:  # type: ignore[override]
        """
        Execute a step in the environment.

        Args:
            action: ResearchAction with action_type and relevant fields.

        Returns:
            ResearchObservation with updated state, reward, and feedback.
        """
        self._state.step_count += 1
        step_num = self._state.step_count

        # Track action
        action_str = f"{action.action_type}:{action.query_terms or action.paper_id or ','.join(action.paper_ids)}"
        self._action_history.append(action_str)

        # --- Validate action type ---
        valid_actions = PHASE_TRANSITIONS.get(self._current_phase, [])
        if action.action_type not in valid_actions:
            reward = -0.1
            self._cumulative_reward += reward
            return self._make_observation(
                reward=reward,
                done=False,
                feedback=f"Invalid action '{action.action_type}' in phase '{self._current_phase}'. Valid: {valid_actions}",
            )

        # --- Execute action ---
        reward = 0.0
        feedback = ""
        done = False

        if action.action_type == "search":
            reward, feedback = self._handle_search(action)
        elif action.action_type == "filter":
            reward, feedback = self._handle_filter(action)
        elif action.action_type == "summarize":
            reward, feedback = self._handle_summarize(action)
        elif action.action_type == "explain":
            reward, feedback = self._handle_explain(action)
        elif action.action_type == "finalize":
            reward, feedback, done = self._handle_finalize()

        # --- Penalize repeated identical actions ---
        if len(self._action_history) >= 2 and self._action_history[-1] == self._action_history[-2]:
            reward *= 0.5
            feedback += " [PENALTY: repeated action, reward halved]"

        # --- Check step limit ---
        if step_num >= self._max_steps and not done:
            done = True
            # Give partial credit based on grader
            partial_score = self._grader.grade(
                self._filtered_paper_ids,
                self._summaries,
                self._explanations,
            )
            partial_score = self._bound_score(partial_score)
            reward = partial_score * 0.5
            feedback += f" [STEP LIMIT REACHED: partial score {partial_score:.2f}]"

        self._cumulative_reward += reward

        return self._make_observation(
            reward=reward,
            done=done,
            feedback=feedback,
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    # -----------------------------------------------------------------
    # Action handlers
    # -----------------------------------------------------------------

    def _handle_search(self, action: ResearchAction) -> tuple:
        """Execute search action."""
        query_terms = action.query_terms.strip()
        if not query_terms:
            return -0.05, "Empty search query. Provide query_terms."

        papers = search_papers(query_terms, top_k=15)
        self._retrieved_paper_ids = [p.id for p in papers]

        reward = self._grader.compute_search_reward(self._retrieved_paper_ids)
        self._current_phase = "filter"

        return reward, f"Found {len(papers)} papers. Proceed to filter the most relevant ones."

    def _handle_filter(self, action: ResearchAction) -> tuple:
        """Execute filter action."""
        paper_ids = action.paper_ids

        if not paper_ids:
            return -0.05, "No paper_ids provided. Select papers to keep."

        # Validate paper IDs exist in retrieved set
        valid_ids = [pid for pid in paper_ids if pid in self._retrieved_paper_ids]
        invalid_ids = [pid for pid in paper_ids if pid not in self._retrieved_paper_ids]

        penalty = 0.0
        hallucinated = []
        if invalid_ids:
            # Check if they're hallucinated (not even in corpus)
            hallucinated = [pid for pid in invalid_ids if get_paper_by_id(pid) is None]
            if hallucinated:
                penalty = -0.15 * len(hallucinated)

        self._filtered_paper_ids = valid_ids
        reward = self._grader.compute_filter_reward(valid_ids) + penalty
        self._current_phase = "summarize"

        feedback_parts = [f"Filtered to {len(valid_ids)} papers."]
        if invalid_ids:
            feedback_parts.append(f"Ignored {len(invalid_ids)} invalid IDs.")
        if hallucinated:
            feedback_parts.append(f"PENALTY: {len(hallucinated)} hallucinated paper IDs.")
        feedback_parts.append("Proceed to summarize key papers.")

        return reward, " ".join(feedback_parts)

    def _handle_summarize(self, action: ResearchAction) -> tuple:
        """Execute summarize action."""
        paper_id = action.paper_id.strip()
        content = action.content.strip()

        if not paper_id:
            return -0.05, "No paper_id provided for summarization."

        if not content:
            return -0.1, "Empty summary content. Provide a substantive summary."

        # Validate paper exists
        paper = get_paper_by_id(paper_id)
        if paper is None:
            return -0.15, f"Paper '{paper_id}' does not exist in corpus. HALLUCINATED ID."

        # Check if paper was in filtered set (warn but allow)
        if paper_id not in self._filtered_paper_ids:
            if paper_id in self._retrieved_paper_ids:
                # Paper was retrieved but not filtered — lower reward
                pass
            else:
                return -0.1, f"Paper '{paper_id}' was not retrieved. Search first."

        self._summaries[paper_id] = content
        reward = self._grader.compute_summary_reward(paper_id, content)

        # Allow transition to explain or continue summarizing
        self._current_phase = "summarize"  # can stay or move to explain

        return reward, f"Summary for '{paper.title[:50]}...' recorded. You can summarize more papers, explain, or finalize."

    def _handle_explain(self, action: ResearchAction) -> tuple:
        """Execute explain action."""
        paper_id = action.paper_id.strip()
        content = action.content.strip()

        if not paper_id:
            return -0.05, "No paper_id provided for explanation."

        if not content:
            return -0.1, "Empty explanation content. Provide a user-friendly explanation."

        paper = get_paper_by_id(paper_id)
        if paper is None:
            return -0.15, f"Paper '{paper_id}' does not exist in corpus. HALLUCINATED ID."

        self._explanations[paper_id] = content
        reward = self._grader.compute_explanation_reward(paper_id, content)
        self._current_phase = "explain"  # can stay or move to finalize

        return reward, f"Explanation for '{paper.title[:50]}...' recorded. You can explain more papers or finalize."

    def _handle_finalize(self) -> tuple:
        """Execute finalize action — compute final composite score."""
        final_score = self._grader.grade(
            self._filtered_paper_ids,
            self._summaries,
            self._explanations,
        )
        final_score = self._bound_score(final_score)

        feedback = (
            f"Episode complete. Final score: {final_score:.3f}. "
            f"Papers filtered: {len(self._filtered_paper_ids)}, "
            f"Summaries: {len(self._summaries)}, "
            f"Explanations: {len(self._explanations)}."
        )

        return final_score, feedback, True

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _make_observation(self, reward: float, done: bool, feedback: str) -> ResearchObservation:
        """Build observation from current state."""
        # Build paper info lists
        retrieved_infos = self._build_paper_infos(self._retrieved_paper_ids)
        filtered_infos = self._build_paper_infos(self._filtered_paper_ids)

        # Determine available actions
        if done:
            available = []
        else:
            available = PHASE_TRANSITIONS.get(self._current_phase, [])

        return ResearchObservation(
            query=self._task.query,
            current_phase=self._current_phase,
            available_actions=available,
            retrieved_papers=retrieved_infos,
            filtered_papers=filtered_infos,
            summaries_so_far=dict(self._summaries),
            explanations_so_far=dict(self._explanations),
            last_action_feedback=feedback,
            step_count=self._state.step_count,
            max_steps=self._max_steps,
            done=done,
            reward=reward,
            metadata={
                "task_name": self._task_name,
                "difficulty": self._task.difficulty,
                "cumulative_reward": self._cumulative_reward,
                "action_history": list(self._action_history),
            },
        )

    def _bound_score(self, score: float) -> float:
        """Ensure returned scores are strictly inside (0, 1)."""
        if score <= 0.0:
            return 0.001
        if score >= 1.0:
            return 0.999
        return score

    @staticmethod
    def _build_paper_infos(paper_ids: List[str]) -> List[PaperInfo]:
        """Convert paper IDs to PaperInfo objects for observation."""
        infos = []
        for pid in paper_ids:
            paper = get_paper_by_id(pid)
            if paper:
                infos.append(PaperInfo(
                    paper_id=paper.id,
                    title=paper.title,
                    abstract_snippet=paper.abstract[:200],
                    keywords=paper.keywords,
                ))
        return infos
