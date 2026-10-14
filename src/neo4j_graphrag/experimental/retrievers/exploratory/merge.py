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

import math
from typing import Literal, Sequence

from .types import (
    ExplorationConfig,
    ExplorerReport,
    KeptItem,
    MergedItem,
    MergedResult,
    ResultGroup,
    Slot,
)
from .verbalizer import Verbalizer


def path_score(probabilities: Sequence[float]) -> float:
    """Geometric mean of the probabilities along a path, so that long paths are not penalized."""
    if not probabilities:
        return 1.0
    logs = [math.log(max(p, 1e-9)) for p in probabilities]
    return math.exp(sum(logs) / len(logs))


def merge_reports(
    reports: Sequence[ExplorerReport],
    slots: Sequence[Slot],
    config: ExplorationConfig,
    verbalizer: Verbalizer,
) -> MergedResult:
    """Merge the reports of independent explorers.

    Items are deduplicated by node, keeping the best path. With
    ``config.only_maximal_paths``, items whose node lies inside the path of
    another item are dropped. Nodes found by several explorers get a bonus.
    At most ``config.max_output_items`` items are kept. Partial reports are
    included; discarded, empty and failed explorers only show up in the
    metadata of the search.

    The items are then presented in groups (:func:`group_items`): with
    ``config.group_paths`` the paths starting from the same node are written as
    one tree, so shared relationships are not repeated. ``verbalizer`` writes
    the text: use :meth:`Verbalizer.for_output`.
    """
    best: dict[str, tuple[float, KeptItem]] = {}
    found_by: dict[str, list[str]] = {}
    slot_probabilities: dict[int, float] = {}
    covered: set[int] = set()

    for report in reports:
        if report.status not in ("sufficient", "partial"):
            continue
        covered.update(report.covered_slots)
        for k, p in report.slot_probabilities.items():
            slot_probabilities[k] = max(p, slot_probabilities.get(k, 0.0))
        for item in report.items:
            node_id = item.node.id
            found_by.setdefault(node_id, []).append(report.seed_id)
            score = path_score(item.step_probabilities)
            if node_id not in best or score > best[node_id][0]:
                best[node_id] = (score, item)

    if config.only_maximal_paths:
        # A node lying inside the path of another result adds nothing: its path
        # is a prefix of that one. Dropping it avoids repeating the same text.
        inner = {n.id for _, item in best.values() for n in item.path_nodes[:-1]}
        best = {k: v for k, v in best.items() if k not in inner}

    merged: list[MergedItem] = []
    for node_id, (score, item) in best.items():
        explorers = found_by[node_id]
        merged.append(
            MergedItem(
                node_id=node_id,
                content=verbalizer.path(item.path_nodes, item.path_relationships),
                score=score + config.multi_explorer_bonus * (len(explorers) - 1),
                explorers=explorers,
                path_probabilities=list(item.step_probabilities),
                path_nodes=list(item.path_nodes),
                path_relationships=list(item.path_relationships),
            )
        )
    merged.sort(key=lambda m: -m.score)
    merged = merged[: config.max_output_items]

    status: Literal["sufficient", "partial", "not_found"]
    if slots and len(covered) == len(slots):
        status = "sufficient"
    elif merged:
        status = "partial"
    else:
        status = "not_found"
    return MergedResult(
        status=status,
        items=merged,
        groups=group_items(merged, config, verbalizer),
        slot_probabilities=slot_probabilities,
        covered_slots=sorted(covered),
    )


def group_items(
    items: Sequence[MergedItem], config: ExplorationConfig, verbalizer: Verbalizer
) -> list[ResultGroup]:
    """Group the items in results, best first.

    With ``config.group_paths`` the items whose paths start from the same node
    form one group, written as a tree; the score of the group is the best score
    of its items. Otherwise every item is a group of its own, written as a path.
    """
    if not config.group_paths:
        return [ResultGroup(i.content, i.score, list(i.explorers), [i]) for i in items]
    by_root: dict[str, list[MergedItem]] = {}
    for item in items:
        by_root.setdefault(item.path_nodes[0].id, []).append(item)
    groups = []
    for members in by_root.values():
        explorers: list[str] = []
        for member in members:
            explorers += [e for e in member.explorers if e not in explorers]
        groups.append(
            ResultGroup(
                content=verbalizer.tree(
                    [(m.path_nodes, m.path_relationships, m.score) for m in members]
                ),
                score=max(m.score for m in members),
                explorers=explorers,
                items=members,
            )
        )
    groups.sort(key=lambda g: -g.score)
    return groups
