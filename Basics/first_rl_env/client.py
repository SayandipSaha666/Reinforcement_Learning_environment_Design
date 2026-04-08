# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Research Paper Assistant Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import PaperInfo, ResearchAction, ResearchObservation


class ResearchAssistantEnv(
    EnvClient[ResearchAction, ResearchObservation, State]
):
    """
    Client for the Research Paper Assistant Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with ResearchAssistantEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.query)
        ...
        ...     result = client.step(ResearchAction(
        ...         action_type="search",
        ...         query_terms="transformer attention NLP"
        ...     ))
        ...     print(len(result.observation.retrieved_papers))

    Example with Docker:
        >>> client = ResearchAssistantEnv.from_docker_image("research_assistant_env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(ResearchAction(action_type="search", query_terms="test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: ResearchAction) -> Dict:
        """
        Convert ResearchAction to JSON payload for step message.

        Args:
            action: ResearchAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
            "query_terms": action.query_terms,
            "paper_ids": action.paper_ids,
            "paper_id": action.paper_id,
            "content": action.content,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ResearchObservation]:
        """
        Parse server response into StepResult[ResearchObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with ResearchObservation
        """
        obs_data = payload.get("observation", {})

        # Parse paper info lists
        retrieved_papers = [
            PaperInfo(**p) for p in obs_data.get("retrieved_papers", [])
        ]
        filtered_papers = [
            PaperInfo(**p) for p in obs_data.get("filtered_papers", [])
        ]

        observation = ResearchObservation(
            query=obs_data.get("query", ""),
            current_phase=obs_data.get("current_phase", "search"),
            available_actions=obs_data.get("available_actions", []),
            retrieved_papers=retrieved_papers,
            filtered_papers=filtered_papers,
            summaries_so_far=obs_data.get("summaries_so_far", {}),
            explanations_so_far=obs_data.get("explanations_so_far", {}),
            last_action_feedback=obs_data.get("last_action_feedback", ""),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 15),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
