# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Research Paper Assistant Environment.

Typed Pydantic models for Action, Observation, and supporting data structures
used in the multi-phase research paper search, filter, summarize, and explain
RL environment.
"""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class PaperInfo(BaseModel):
    """Lightweight paper representation visible to the agent."""

    paper_id: str = Field(default="", description="Unique paper identifier")
    title: str = Field(default="", description="Paper title")
    abstract_snippet: str = Field(
        default="", description="First 200 chars of abstract"
    )
    keywords: List[str] = Field(
        default_factory=list, description="Paper keywords"
    )


class ResearchAction(Action):
    """
    Action for the Research Paper Assistant environment.

    The agent must choose an action_type matching the current phase:
      - search:    provide query_terms to search the corpus
      - filter:    provide paper_ids to keep from retrieved set
      - summarize: provide paper_id + content (the agent's summary)
      - explain:   provide paper_id + content (the agent's explanation)
      - finalize:  signal episode completion
    """

    action_type: str = Field(
        ...,
        description="One of: search, filter, summarize, explain, finalize",
    )
    query_terms: str = Field(
        default="",
        description="Search query terms (for action_type='search')",
    )
    paper_ids: List[str] = Field(
        default_factory=list,
        description="Paper IDs to keep (for action_type='filter')",
    )
    paper_id: str = Field(
        default="",
        description="Target paper ID (for summarize/explain)",
    )
    content: str = Field(
        default="",
        description="Agent-generated summary or explanation text",
    )


class ResearchObservation(Observation):
    """
    Observation from the Research Paper Assistant environment.

    Provides the agent with current state information and feedback.
    """

    query: str = Field(default="", description="The research query for this episode")
    current_phase: str = Field(
        default="search",
        description="Current phase: search, filter, summarize, explain, finalize",
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="Valid action_type values for the current phase",
    )
    retrieved_papers: List[PaperInfo] = Field(
        default_factory=list,
        description="Papers found by search",
    )
    filtered_papers: List[PaperInfo] = Field(
        default_factory=list,
        description="Papers after filtering",
    )
    summaries_so_far: Dict[str, str] = Field(
        default_factory=dict,
        description="paper_id -> agent summary generated so far",
    )
    explanations_so_far: Dict[str, str] = Field(
        default_factory=dict,
        description="paper_id -> agent explanation generated so far",
    )
    last_action_feedback: str = Field(
        default="",
        description="Environment feedback on the last action",
    )
    step_count: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=15, description="Maximum allowed steps")
