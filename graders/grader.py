"""
Deterministic Grader for the Research Paper Assistant Environment
==================================================================

Computes a final score in (0, 1) based on four metrics:
  - relevance:            F1 of filtered papers vs ground truth
  - correctness:          keyword overlap of agent summaries vs reference keywords
  - completeness:         sub-topic coverage + minimum summary/explanation counts
  - explanation_quality:  readability + accuracy + synthesis (cross-references)

All computation is deterministic — no randomness, no external API calls.
Uses regex-based tokenization (no NLTK dependency).

SCORE VALIDATION: OpenEnv requires ALL task scores to be strictly inside (0, 1).
Score must be > 0.0 AND < 1.0. Values of exactly 0.0 or 1.0 are INVALID.
All scores are bounded using _safe_bound() to guarantee strict open-interval compliance.
"""

import math
import re
from typing import Dict, List, Set

from data.paper_corpus import get_paper_by_id


# Safety epsilon: ensures final scores are strictly inside (0, 1)
# Using 1e-6 gives 6 orders of magnitude of breathing room from 0 and 1,
# while remaining indistinguishable from "perfect" scores in practice.
_SCORE_EPSILON = 1e-6


def _safe_bound(score: float) -> float:
    """
    Bound a score to the strict open interval (0, 1).

    This is the SINGLE SOURCE OF TRUTH for all score normalization.
    Every score — final composites and intermediate metrics — passes
    through here before being returned or used in further computation.

    Args:
        score: Raw score from any computation.

    Returns:
        A float strictly inside (0, 1). Guarantees:
        - score > 0.0  (never 0.0, never negative)
        - score < 1.0  (never 1.0, never above)

    Floating-point edge cases handled:
        - NaN               → _SCORE_EPSILON
        - +inf              → 1.0 - _SCORE_EPSILON
        - -inf              → _SCORE_EPSILON
        - Exactly 0.0       → _SCORE_EPSILON
        - Exactly 1.0       → 1.0 - _SCORE_EPSILON
        - In (0, 1)         → unchanged
        - Outside [0, 1]    → clamped to open interval
    """
    # Handle non-finite values first
    if not math.isfinite(score):
        return _SCORE_EPSILON if math.isnan(score) or score < 0 else (1.0 - _SCORE_EPSILON)

    # Strict lower bound: never 0.0
    if score <= 0.0:
        return _SCORE_EPSILON

    # Strict upper bound: never 1.0
    if score >= 1.0:
        return 1.0 - _SCORE_EPSILON

    return score


class ResearchGrader:
    """Deterministic grader for research assistant episodes."""

    def __init__(self, task_config):
        """
        Args:
            task_config: A TaskConfig instance with ground truth and weights.
        """
        self.task = task_config

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def grade(
        self,
        filtered_paper_ids: List[str],
        summaries: Dict[str, str],
        explanations: Dict[str, str],
    ) -> float:
        """
        Compute final composite score in [0.0, 1.0].

        Args:
            filtered_paper_ids: Paper IDs the agent selected via filter action.
            summaries: dict of paper_id -> agent-generated summary text.
            explanations: dict of paper_id -> agent-generated explanation text.

        Returns:
            Weighted composite score.
        """
        rel = _safe_bound(self.compute_relevance(filtered_paper_ids))
        corr = _safe_bound(self.compute_correctness(summaries))
        comp = _safe_bound(self.compute_completeness(summaries, explanations))
        expl = _safe_bound(self.compute_explanation_quality(explanations))

        w = self.task.grading_weights
        raw_score = (
            w["relevance"] * rel
            + w["correctness"] * corr
            + w["completeness"] * comp
            + w["explanation_quality"] * expl
        )

        # Final bounding: the only line that matters for validator compliance
        return _safe_bound(raw_score)

    # -----------------------------------------------------------------
    # Individual metrics
    # -----------------------------------------------------------------

    def compute_relevance(self, filtered_paper_ids: List[str]) -> float:
        """
        F1 score of filtered papers vs ground truth.

        relevance = 2 * precision * recall / (precision + recall + ε)
        """
        gt = set(self.task.ground_truth_paper_ids)
        filtered = set(filtered_paper_ids)

        if not filtered:
            return 0.0

        tp = len(gt & filtered)
        precision = tp / len(filtered) if filtered else 0.0
        recall = tp / len(gt) if gt else 0.0

        if precision + recall == 0:
            return 0.0

        return 2.0 * precision * recall / (precision + recall)

    def compute_correctness(self, summaries: Dict[str, str]) -> float:
        """
        Average keyword overlap between agent summaries and reference keywords.

        For each summarized paper that has reference keywords, compute:
            hit_rate = |agent_words ∩ ref_keywords| / |ref_keywords|
        Return mean across all summarized papers with reference data.
        """
        if not summaries:
            return 0.0

        scores = []
        for paper_id, summary_text in summaries.items():
            ref_keywords = self.task.reference_keywords.get(paper_id, [])
            if not ref_keywords:
                continue

            agent_words = self._tokenize(summary_text.lower())
            hits = sum(
                1 for kw in ref_keywords
                if self._keyword_in_text(kw.lower(), agent_words, summary_text.lower())
            )
            scores.append(hits / len(ref_keywords))

        return sum(scores) / len(scores) if scores else 0.0

    def compute_completeness(
        self,
        summaries: Dict[str, str],
        explanations: Dict[str, str],
    ) -> float:
        """
        Measures how many required sub-topics the agent covered and
        whether it met minimum summary/explanation counts.

        completeness = 0.6 * sub_topic_coverage + 0.2 * summary_count_ratio + 0.2 * explanation_count_ratio
        """
        # Sub-topic coverage
        covered_topics: Set[str] = set()
        all_text_paper_ids = set(summaries.keys()) | set(explanations.keys())

        for paper_id in all_text_paper_ids:
            paper = get_paper_by_id(paper_id)
            if paper and paper.sub_topic in self.task.required_sub_topics:
                covered_topics.add(paper.sub_topic)

        topic_coverage = (
            len(covered_topics) / len(self.task.required_sub_topics)
            if self.task.required_sub_topics
            else 1.0
        )

        # Summary count ratio
        summary_ratio = min(len(summaries) / self.task.min_summaries, 1.0) if self.task.min_summaries > 0 else 1.0

        # Explanation count ratio
        explanation_ratio = min(len(explanations) / self.task.min_explanations, 1.0) if self.task.min_explanations > 0 else 1.0

        return 0.6 * topic_coverage + 0.2 * summary_ratio + 0.2 * explanation_ratio

    def compute_explanation_quality(self, explanations: Dict[str, str]) -> float:
        """
        Composite of readability + accuracy + synthesis.

        explanation_quality = 0.3 * readability + 0.4 * accuracy + 0.3 * synthesis
        """
        if not explanations:
            return 0.0

        readability_scores = []
        accuracy_scores = []

        for paper_id, explanation_text in explanations.items():
            # Readability: average sentence length < 30 words is good
            readability_scores.append(self._compute_readability(explanation_text))

            # Accuracy: keyword overlap with explanation reference
            expl_keywords = self.task.explanation_keywords.get(paper_id, [])
            if expl_keywords:
                agent_words = self._tokenize(explanation_text.lower())
                hits = sum(
                    1 for kw in expl_keywords
                    if self._keyword_in_text(kw.lower(), agent_words, explanation_text.lower())
                )
                accuracy_scores.append(hits / len(expl_keywords))

        readability = sum(readability_scores) / len(readability_scores) if readability_scores else 0.0
        accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0

        # Synthesis: does the explanation reference multiple papers?
        synthesis = 0.5  # default
        if self.task.cross_reference_required:
            # Check if any explanation mentions concepts from other papers
            synthesis = self._compute_synthesis(explanations)
        else:
            # For easy tasks, just being present counts
            synthesis = 1.0 if explanations else 0.0

        return 0.3 * readability + 0.4 * accuracy + 0.3 * synthesis

    # -----------------------------------------------------------------
    # Step-level reward helpers (used by environment)
    # -----------------------------------------------------------------

    def compute_search_reward(self, retrieved_paper_ids: List[str]) -> float:
        """
        Reward for search action: proportion of ground truth papers
        that appear in the retrieved set.

        R = 0.1 * |retrieved ∩ gt| / |gt|
        """
        gt = set(self.task.ground_truth_paper_ids)
        retrieved = set(retrieved_paper_ids)
        overlap = len(gt & retrieved)
        return _safe_bound(0.1 * (overlap / len(gt)) if gt else 0.0)

    def compute_filter_reward(self, filtered_paper_ids: List[str]) -> float:
        """
        Reward for filter action: F1 scaled to [0, 0.15].

        R = 0.15 * F1(filtered, gt)
        """
        return _safe_bound(0.15 * self.compute_relevance(filtered_paper_ids))

    def compute_summary_reward(self, paper_id: str, summary_text: str) -> float:
        """
        Reward for summarize action: keyword overlap scaled to [0, 0.2].

        R = 0.2 * keyword_hit_rate
        """
        ref_keywords = self.task.reference_keywords.get(paper_id, [])
        if not ref_keywords:
            return _safe_bound(0.05)  # small reward for summarizing any paper

        agent_words = self._tokenize(summary_text.lower())
        hits = sum(
            1 for kw in ref_keywords
            if self._keyword_in_text(kw.lower(), agent_words, summary_text.lower())
        )
        return _safe_bound(0.2 * (hits / len(ref_keywords)))

    def compute_explanation_reward(self, paper_id: str, explanation_text: str) -> float:
        """
        Reward for explain action: quality scaled to [0, 0.2].

        R = 0.2 * (0.5 * readability + 0.5 * accuracy)
        """
        readability = self._compute_readability(explanation_text)

        expl_keywords = self.task.explanation_keywords.get(paper_id, [])
        if not expl_keywords:
            return _safe_bound(0.1 * readability)  # partial credit

        agent_words = self._tokenize(explanation_text.lower())
        hits = sum(
            1 for kw in expl_keywords
            if self._keyword_in_text(kw.lower(), agent_words, explanation_text.lower())
        )
        accuracy = hits / len(expl_keywords)

        return _safe_bound(0.2 * (0.5 * readability + 0.5 * accuracy))

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """Simple regex-based word tokenization."""
        return set(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower()))

    @staticmethod
    def _keyword_in_text(keyword: str, word_set: Set[str], full_text: str) -> bool:
        """
        Check if keyword appears in text.
        Handles multi-word keywords by checking substring in full text.
        Single-word keywords checked against word set for speed.
        """
        if " " in keyword or "-" in keyword:
            # Multi-word keyword: check as substring
            return keyword in full_text
        return keyword in word_set

    @staticmethod
    def _compute_readability(text: str) -> float:
        """
        Simple readability heuristic based on sentence length.
        Returns score in [0, 1].

        - Average sentence length < 25 words: 1.0
        - 25-40 words: linear decay
        - > 40 words: 0.2
        """
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)

        if avg_len <= 25:
            return 1.0
        elif avg_len <= 40:
            return 1.0 - 0.8 * ((avg_len - 25) / 15)
        else:
            return 0.2

    def _compute_synthesis(self, explanations: Dict[str, str]) -> float:
        """
        Check if explanations reference concepts from multiple paper domains.
        Returns score in [0, 1] based on cross-topic keyword overlap.
        """
        if len(explanations) < 2:
            return 0.3

        # Collect all unique sub-topics from explained papers
        topics_explained = set()
        for paper_id in explanations:
            paper = get_paper_by_id(paper_id)
            if paper:
                topics_explained.add(paper.sub_topic)

        # Check if any single explanation mentions keywords from other topics' papers
        cross_refs = 0
        total_checks = 0

        for paper_id, expl_text in explanations.items():
            expl_lower = expl_text.lower()
            expl_paper = get_paper_by_id(paper_id)
            if not expl_paper:
                continue

            # Check for keywords from papers in OTHER sub-topics
            for other_pid in explanations:
                if other_pid == paper_id:
                    continue
                other_paper = get_paper_by_id(other_pid)
                if not other_paper or other_paper.sub_topic == expl_paper.sub_topic:
                    continue

                total_checks += 1
                # Check if explanation mentions the other paper's key concepts
                other_keywords = self.task.explanation_keywords.get(other_pid, [])
                if any(kw.lower() in expl_lower for kw in other_keywords[:3]):
                    cross_refs += 1

        if total_checks == 0:
            return 0.5

        return min(1.0, 0.3 + 0.7 * (cross_refs / total_checks))
