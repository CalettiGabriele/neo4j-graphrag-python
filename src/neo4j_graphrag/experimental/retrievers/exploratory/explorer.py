#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Sequence

from neo4j_graphrag.experimental.system_one import (
    Answer,
    ChoiceAnswer,
    PredicateAnswer,
    Question,
    RefusalAnswer,
    SystemOneModelInterface,
)

from .graph_access import GraphAccess
from .prompts import ACTION_DISCARD, ACTION_QUESTION, MissionTemplates
from .types import (
    CandidatePath,
    ExplorationConfig,
    ExplorerReport,
    ExplorerStatus,
    GraphNode,
    KeptItem,
    RelationshipTypeSummary,
    Slot,
)
from .verbalizer import Verbalizer

logger = logging.getLogger(__name__)

AnswerGetter = Callable[[str], Optional[Answer]]


class Explorer:
    """Explores the graph from one seed node. Independent from any other explorer.

    At every turn the explorer reads the candidates around its frontier and sends
    **one** request to the System One Model, with one question for how to proceed,
    one for every candidate (or, for hubs, for every relationship group) and one
    for every information requirement (slot). The model only provides
    probabilities: the decisions (continue, stop and discard, stop and return)
    are taken here, by comparing them with the thresholds of the configuration.

    Notes on the behavior:

    - A direct neighbor that is rejected is never offered again by this explorer,
      even if it later becomes relevant through another branch: this bounds the
      cost. A node rejected while seen through a chain stays eligible, because it
      may be reached again once the intermediate node is kept.
    - The neighbors of a hub are read page after page (ordered by element id),
      as long as its relationship types stay approved and turns are left.
    - Every truncation (limits per node, per turn, or on the number of questions)
      is counted in the report.

    An explorer is single-use: create a new one for every search.

    Args:
        seed (GraphNode): Node the exploration starts from.
        query_text (str): The user question.
        slots (Sequence[Slot]): Information requirements the context must satisfy.
        config (ExplorationConfig): Budget and thresholds.
        graph (GraphAccess): Access to the graph.
        model (SystemOneModelInterface): The System One Model that judges the branches.
        verbalizer (Verbalizer): Writes nodes and paths for the model.
        templates (MissionTemplates): Wording of the questions.
        deadline (Optional[float]): ``time.monotonic()`` value after which no new
            turn is started.
    """

    def __init__(
        self,
        seed: GraphNode,
        query_text: str,
        slots: Sequence[Slot],
        config: ExplorationConfig,
        graph: GraphAccess,
        model: SystemOneModelInterface,
        verbalizer: Verbalizer,
        templates: MissionTemplates,
        deadline: Optional[float] = None,
    ) -> None:
        self.seed = seed
        self.query_text = query_text
        self.slots = list(slots)
        self.config = config
        self.graph = graph
        self.model = model
        self.verbalizer = verbalizer
        self.templates = templates
        self.deadline = deadline

        self.kept: dict[str, KeptItem] = {
            seed.id: KeptItem(
                node=seed,
                path_nodes=[seed],
                path_relationships=[],
                segment_nodes=[seed],
            )
        }
        self.visited: set[str] = {seed.id}
        self.frontier: list[str] = [seed.id]
        self.approved_types: dict[str, list[str]] = {}
        self.hub_cursors: dict[str, str] = {}
        self.covered: set[int] = set()
        self.slot_probabilities: dict[int, float] = {}
        self.turns = 0
        self.depth = 0
        self.questions = 0
        self.input_tokens = 0
        self.truncations = 0
        self.timed_out = False
        self.trace: list[dict[str, Any]] = []
        self._turn_scores: list[tuple[float, CandidatePath]] = []

    # -- public ---------------------------------------------------------------

    def run(self) -> ExplorerReport:
        """Run the exploration. Never raises: a failure becomes a report with
        status ``error``, which keeps the exception for the caller."""
        try:
            return self._run()
        except Exception as e:
            logger.debug("Explorer for seed %s failed", self.seed.id, exc_info=True)
            return self._report("error", error=f"{type(e).__name__}: {e}", exception=e)

    # -- exploration ------------------------------------------------------------

    def _run(self) -> ExplorerReport:
        cfg = self.config
        for turn in range(1, cfg.max_turns + 1):
            if self.deadline is not None and time.monotonic() >= self.deadline:
                self.timed_out = True
                self.trace.append({"turn": turn, "decision": "timeout"})
                break
            self.turns = turn
            paths, summaries = self._gather()
            if not paths and not summaries:
                self.frontier = []
                self.trace.append({"turn": turn, "candidates": 0, "hubs": 0})
                break
            paths, summaries = self._fit_budget(paths, summaries)

            questions = self._build_questions(paths, summaries)
            response = self.model.decide(
                self._render_input(paths, summaries), questions
            )
            self.questions += len(questions)
            if response.usage and response.usage.input_tokens:
                self.input_tokens += response.usage.input_tokens

            discard_requested = self._action(response.get(ACTION_QUESTION))
            added = self._apply_candidates(paths, response.get)
            self._apply_hubs(summaries, response.get)
            self._apply_slots(response.get)

            decision = self._decide(discard_requested)
            if decision == "sufficient":
                added += self._keep_evidence()
            self.trace.append(
                {
                    "turn": turn,
                    "candidates": len(paths),
                    "hubs": len(summaries),
                    "kept_added": added,
                    "discard_requested": discard_requested,
                    "slots": dict(self.slot_probabilities),
                    "decision": decision or "continue",
                }
            )
            if decision is not None:
                return self._report(decision)
        return self._report(self._exhausted_status())

    def _gather(
        self,
    ) -> tuple[list[CandidatePath], list[RelationshipTypeSummary]]:
        cfg = self.config
        exclude = list(self.visited)
        degrees = self.graph.degrees(
            [n for n in self.frontier if n not in self.approved_types], exclude
        )
        typed = [n for n in self.frontier if n in self.approved_types]
        plain = [
            n
            for n in self.frontier
            if n not in self.approved_types and degrees.get(n, 0) <= cfg.max_candidates
        ]
        hubs = [
            n
            for n in self.frontier
            if n not in self.approved_types and degrees.get(n, 0) > cfg.max_candidates
        ]
        paths = self.graph.fetch_paths(
            plain, cfg.hops_per_turn, exclude, cfg.max_candidates
        )
        per_origin: dict[str, int] = {}
        for path in paths:
            per_origin[path.origin_id] = per_origin.get(path.origin_id, 0) + 1
        self.truncations += sum(
            1 for count in per_origin.values() if count >= cfg.max_candidates
        )
        for node_id in typed:
            page = self.graph.fetch_paths(
                [node_id],
                cfg.hops_per_turn,
                exclude,
                cfg.max_candidates,
                first_hop_types=self.approved_types[node_id],
                after=self.hub_cursors.get(node_id),
            )
            paths += page
            if len(page) >= cfg.max_candidates:
                # More neighbors of the approved types: read the next page later.
                self.hub_cursors[node_id] = page[-1].end.id
            else:
                del self.approved_types[node_id]
                self.hub_cursors.pop(node_id, None)
        summaries: list[RelationshipTypeSummary] = []
        for node_id in hubs:
            summaries += self.graph.relationship_type_summary(node_id, exclude)
        unique: dict[str, CandidatePath] = {}
        for path in paths:
            unique.setdefault(path.end.id, path)
        return list(unique.values()), summaries

    def _fit_budget(
        self,
        paths: list[CandidatePath],
        summaries: list[RelationshipTypeSummary],
    ) -> tuple[list[CandidatePath], list[RelationshipTypeSummary]]:
        """Keep the number of questions of the turn within the configured limits."""
        cfg = self.config
        budget = max(cfg.max_questions_per_turn - 1 - len(self.slots), 1)
        max_paths = min(cfg.max_candidates_per_turn, budget)
        # Hub groups get what candidates leave, but at least a quarter of the budget.
        max_groups = max(budget - min(len(paths), max_paths), budget // 4)
        kept_summaries = summaries[:max_groups]
        kept_paths = paths[: min(max_paths, budget - len(kept_summaries))]
        if len(kept_paths) < len(paths) or len(kept_summaries) < len(summaries):
            self.truncations += 1
        return kept_paths, kept_summaries

    def _build_questions(
        self,
        paths: Sequence[CandidatePath],
        summaries: Sequence[RelationshipTypeSummary],
    ) -> list[Question]:
        t = self.templates
        questions: list[Question] = [t.action_question()]
        questions += [t.keep_question(i) for i in range(len(paths))]
        questions += [t.type_question(j) for j in range(len(summaries))]
        questions += [t.slot_question(k, s) for k, s in enumerate(self.slots)]
        return questions

    def _render_input(
        self,
        paths: Sequence[CandidatePath],
        summaries: Sequence[RelationshipTypeSummary],
    ) -> str:
        v = self.verbalizer
        # Each kept node adds only its own segment: every relationship appears once.
        kept = [
            v.path(i.segment_nodes, i.segment_relationships) for i in self.kept.values()
        ]
        candidates = [v.path(p.nodes, p.relationships) for p in paths]
        hubs = [
            f"{v.node(self.kept[s.node_id].node)} -> {v.type_summary(s)}"
            for s in summaries
        ]
        return self.templates.render_input(
            self.query_text, self.slots, sorted(self.covered), kept, candidates, hubs
        )

    # -- applying the answers ------------------------------------------------------

    @staticmethod
    def _probability(answer: Optional[Answer]) -> float:
        # A refusal is treated as "no evidence".
        return answer.probability if isinstance(answer, PredicateAnswer) else 0.0

    def _action(self, answer: Optional[Answer]) -> bool:
        """Whether the model asks to stop and discard the exploration."""
        if isinstance(answer, ChoiceAnswer):
            if answer.choice != ACTION_DISCARD:
                return False
            return (
                answer.probabilities.get(ACTION_DISCARD, 1.0)
                >= self.config.discard_threshold
            )
        if isinstance(answer, RefusalAnswer):
            logger.debug("Refusal on the action question: %s", answer.reason)
        return True

    def _keep(self, path: CandidatePath, probability: float) -> None:
        parent = self.kept[path.origin_id]
        self.kept[path.end.id] = KeptItem(
            node=path.end,
            path_nodes=parent.path_nodes + path.nodes[1:],
            path_relationships=parent.path_relationships + path.relationships,
            step_probabilities=parent.step_probabilities + [probability],
            segment_nodes=list(path.nodes),
            segment_relationships=list(path.relationships),
        )

    def _apply_candidates(
        self, paths: Sequence[CandidatePath], get: AnswerGetter
    ) -> int:
        self._turn_scores = []
        accepted: list[tuple[float, CandidatePath]] = []
        for i, path in enumerate(paths):
            parent_depth = len(self.kept[path.origin_id].path_relationships)
            self.depth = max(self.depth, parent_depth + len(path.relationships))
            if len(path.relationships) == 1:
                # A rejected direct neighbor is not offered again (see class notes).
                self.visited.add(path.end.id)
            p = self._probability(get(f"keep_{i}"))
            self._turn_scores.append((p, path))
            if p >= self.config.keep_threshold:
                accepted.append((p, path))
        accepted.sort(key=lambda x: -x[0])
        capacity = max(self.config.max_nodes - len(self.kept), 0)
        if len(accepted) > capacity:
            self.truncations += 1
        new_frontier: list[str] = []
        for p, path in accepted[:capacity]:
            self._keep(path, p)
            self.visited.add(path.end.id)
            new_frontier.append(path.end.id)
        self.frontier = new_frontier
        return len(new_frontier)

    def _keep_evidence(self) -> int:
        """When the slots are covered, also keep the candidates of the last turn
        that are likely to be the evidence, even if below the keep threshold:
        the coverage was judged on them too."""
        added = 0
        for p, path in sorted(self._turn_scores, key=lambda x: -x[0]):
            if p >= self.config.evidence_threshold and path.end.id not in self.kept:
                self._keep(path, p)
                added += 1
        return added

    def _apply_hubs(
        self, summaries: Sequence[RelationshipTypeSummary], get: AnswerGetter
    ) -> None:
        for j, summary in enumerate(summaries):
            parent_depth = len(self.kept[summary.node_id].path_relationships)
            self.depth = max(self.depth, parent_depth + 1)
            if self._probability(get(f"type_{j}")) >= self.config.keep_threshold:
                self.approved_types.setdefault(summary.node_id, []).append(summary.type)
        for node_id in self.approved_types:
            if node_id not in self.frontier:
                self.frontier.append(node_id)

    def _apply_slots(self, get: AnswerGetter) -> None:
        cfg = self.config
        for k in range(len(self.slots)):
            p = self._probability(get(f"slot_{k}"))
            self.slot_probabilities[k] = p
            if k in self.covered:
                if p < cfg.slot_reopen_threshold:
                    self.covered.discard(k)
            elif p >= cfg.slot_covered_threshold:
                self.covered.add(k)

    # -- decisions -------------------------------------------------------------------

    def _decide(self, discard_requested: bool) -> Optional[ExplorerStatus]:
        """Decide whether to stop. ``None`` means: continue with another turn."""
        cfg = self.config
        if self.slots and len(self.covered) == len(self.slots):
            return "sufficient"
        if (
            discard_requested
            and self.depth >= cfg.min_hops_before_discard
            and len(self.kept) <= 1
        ):
            return "discarded"
        if not self.frontier or len(self.kept) >= cfg.max_nodes:
            return self._exhausted_status()
        return None

    def _exhausted_status(self) -> ExplorerStatus:
        return "partial" if len(self.kept) > 1 else "not_found"

    def _report(
        self,
        status: ExplorerStatus,
        error: Optional[str] = None,
        exception: Optional[BaseException] = None,
    ) -> ExplorerReport:
        items: list[KeptItem] = []
        if status in ("sufficient", "partial"):
            items = list(self.kept.values())
            if len(items) > 1:
                items = items[1:]  # the seed is already part of every path
        return ExplorerReport(
            seed_id=self.seed.id,
            status=status,
            items=items,
            slot_probabilities=dict(self.slot_probabilities),
            covered_slots=sorted(self.covered),
            turns=self.turns,
            hops_explored=self.depth,
            questions=self.questions,
            input_tokens=self.input_tokens,
            timed_out=self.timed_out,
            truncations=self.truncations,
            error=error,
            exception=exception,
            trace=list(self.trace),
        )
