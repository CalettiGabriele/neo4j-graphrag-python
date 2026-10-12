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
"""Test doubles: an in-memory graph and a scripted System One Model."""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from neo4j_graphrag.experimental.retrievers.exploratory.types import (
    CandidatePath,
    GraphNode,
    GraphRelationship,
    RelationshipTypeSummary,
)
from neo4j_graphrag.experimental.system_one import (
    ChoiceAnswer,
    DecisionResponse,
    PredicateAnswer,
    Question,
    RefusalAnswer,
    SystemOneModelInterface,
    SystemOneUsage,
)


def node(node_id: str, label: str = "Company", **properties: Any) -> GraphNode:
    properties.setdefault("name", node_id)
    return GraphNode(id=node_id, labels=[label], properties=properties)


class FakeGraph:
    """In-memory stand-in for GraphAccess, following the same rules: paths are
    ordered by end node id, excluded ids are neither returned nor crossed,
    ``after`` reads the next page, and relationships are traversed both ways."""

    def __init__(
        self,
        nodes: Sequence[GraphNode],
        relationships: Sequence[tuple[str, str, str]],
        degree_overrides: Optional[dict[str, int]] = None,
    ) -> None:
        self.nodes = {n.id: n for n in nodes}
        self.rels = [
            GraphRelationship(id=f"r{i}", type=t, start_id=a, end_id=b)
            for i, (a, t, b) in enumerate(relationships)
        ]
        self.degree_overrides = degree_overrides or {}
        self.fetch_calls: list[dict[str, Any]] = []
        self.summary_calls: list[str] = []

    def _adjacent(self, node_id: str) -> list[tuple[GraphRelationship, str]]:
        out = []
        for r in self.rels:
            if r.start_id == node_id:
                out.append((r, r.end_id))
            elif r.end_id == node_id:
                out.append((r, r.start_id))
        return out

    def node_details(self, node_ids: Sequence[str]) -> list[GraphNode]:
        return [self.nodes[i] for i in node_ids if i in self.nodes]

    def degrees(
        self, node_ids: Sequence[str], exclude_ids: Sequence[str] = ()
    ) -> dict[str, int]:
        return {
            i: self.degree_overrides.get(
                i, sum(1 for _, o in self._adjacent(i) if o not in exclude_ids)
            )
            for i in node_ids
        }

    def fetch_paths(
        self,
        origin_ids: Sequence[str],
        hops: int,
        exclude_ids: Sequence[str],
        limit: int,
        first_hop_types: Optional[Sequence[str]] = None,
        after: Optional[str] = None,
    ) -> list[CandidatePath]:
        self.fetch_calls.append(
            {
                "origins": list(origin_ids),
                "hops": hops,
                "first_hop_types": first_hop_types,
                "after": after,
            }
        )
        result: list[CandidatePath] = []
        for origin in origin_ids:
            found: list[CandidatePath] = []

            def walk(
                current: str, nodes: list[str], rels: list[GraphRelationship]
            ) -> None:
                if rels and current not in exclude_ids and current != origin:
                    if not any(i in exclude_ids for i in nodes[1:-1]):
                        found.append(
                            CandidatePath(
                                origin_id=origin,
                                nodes=[self.nodes[i] for i in nodes],
                                relationships=list(rels),
                            )
                        )
                if len(rels) == hops:
                    return
                for rel, other in self._adjacent(current):
                    if rel in rels or other in nodes:
                        continue
                    if first_hop_types and not rels and rel.type not in first_hop_types:
                        continue
                    walk(other, nodes + [other], rels + [rel])

            walk(origin, [origin], [])
            found.sort(key=lambda p: (p.end.id, len(p.relationships)))
            if after is not None:
                found = [p for p in found if p.end.id > after]
            result += found[:limit]
        return result

    def relationship_type_summary(
        self, node_id: str, exclude_ids: Sequence[str] = ()
    ) -> list[RelationshipTypeSummary]:
        self.summary_calls.append(node_id)
        groups: dict[tuple[str, str], list[GraphNode]] = {}
        for rel, other in self._adjacent(node_id):
            if other in exclude_ids:
                continue
            direction = "out" if rel.start_id == node_id else "in"
            groups.setdefault((rel.type, direction), []).append(self.nodes[other])
        return [
            RelationshipTypeSummary(
                node_id=node_id,
                type=t,
                direction=d,  # type: ignore[arg-type]
                count=len(ns),
                examples=sorted(ns, key=lambda n: n.id)[:3],
            )
            for (t, d), ns in sorted(groups.items(), key=lambda g: -len(g[1]))
        ]

    def chunks_to_entities(
        self, chunk_ids: Sequence[str], relationship_type: str, limit: int
    ) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for rel in self.rels:
            if rel.type == relationship_type and rel.end_id in chunk_ids:
                result.setdefault(rel.end_id, []).append(rel.start_id)
        return {k: sorted(v)[:limit] for k, v in result.items()}


_NODE = re.compile(r'[A-Z][A-Za-z:]* "([^"]+)"')


def candidate_names(input: str) -> dict[int, str]:
    """Index of every candidate of an input -> name of its last node."""
    result = {}
    for index, line in re.findall(r"^\[(\d+)\] (.*)$", input, flags=re.M):
        names = _NODE.findall(line)
        if names:
            result[int(index)] = names[-1]
    return result


def group_types(input: str) -> dict[int, str]:
    """Index of every hub group of an input -> relationship type."""
    return {
        int(i): t
        for i, t in re.findall(r"^\[H(\d+)\] .* -> (\S+) \(", input, flags=re.M)
    }


Turn = dict[str, Any]


class ScriptedModel(SystemOneModelInterface):
    """Answers turn after turn following a script.

    A turn maps a key to a value:

    - ``"action"``: a choice (string, or ``(choice, probability)``), default ``continue``;
    - ``"keep:<name>"``: probability of keeping the candidate whose last node is ``name``;
    - ``"type:<TYPE>"``: probability of following the hub groups of that type;
    - ``"slot_<k>"`` (or any question name): probability of that predicate;
    - the value ``"refusal"`` produces a refusal.

    Unmentioned predicates get 0.0. After the script is over the last turn is repeated.
    """

    def __init__(self, script: Sequence[Turn]) -> None:
        super().__init__("scripted")
        self.script = list(script)
        self.requests: list[tuple[str, list[Question]]] = []

    def _decide(self, input: str, questions: Sequence[Question]) -> DecisionResponse:
        self.requests.append((input, list(questions)))
        turn = self.script[min(len(self.requests), len(self.script)) - 1]
        names, types = candidate_names(input), group_types(input)
        answers: list[Any] = []
        for q in questions:
            value = turn.get(q.name)
            if q.name.startswith("keep_") and value is None:
                value = turn.get(f"keep:{names.get(int(q.name[5:]))}")
            if q.name.startswith("type_") and value is None:
                value = turn.get(f"type:{types.get(int(q.name[5:]))}")
            if value == "refusal":
                answers.append(RefusalAnswer(name=q.name))
            elif q.name == "action":
                choice, p = (
                    value if isinstance(value, tuple) else (value or "continue", 0.9)
                )
                answers.append(
                    ChoiceAnswer(
                        name="action", choice=choice, probabilities={choice: p}
                    )
                )
            else:
                answers.append(PredicateAnswer(name=q.name, probability=value or 0.0))
        return DecisionResponse(answers=answers, usage=SystemOneUsage(input_tokens=100))


class RaisingModel(SystemOneModelInterface):
    def __init__(self, error: Exception) -> None:
        super().__init__("raising")
        self.error = error

    def _decide(self, input: str, questions: Sequence[Question]) -> DecisionResponse:
        raise self.error


def acme_graph() -> FakeGraph:
    """ACME's main customer is Beta, which has suppliers; one of them was sanctioned."""
    nodes = [
        node("acme"),
        node("torino", "City"),
        node("automotive", "Sector"),
        node("beta"),
        node("gamma"),
        node("delta"),
        node("agcm", "Authority"),
    ]
    rels = [
        ("acme", "sede_legale", "torino"),
        ("acme", "settore", "automotive"),
        ("acme", "cliente_principale", "beta"),
        ("beta", "fornito_da", "gamma"),
        ("beta", "fornito_da", "delta"),
        ("gamma", "sanzionata_da", "agcm"),
    ]
    return FakeGraph(nodes, rels)
