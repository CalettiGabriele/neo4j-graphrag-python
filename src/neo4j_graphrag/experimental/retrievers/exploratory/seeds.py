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

import inspect
from dataclasses import dataclass, field
import logging
from typing import Any, Sequence

from neo4j_graphrag.components.types import LexicalGraphConfig
from neo4j_graphrag.retrievers.base import Retriever

from .exceptions import ExploratoryRetrieverError
from .graph_access import GraphAccess
from .types import ExplorationConfig, GraphNode

logger = logging.getLogger(__name__)


@dataclass
class SeedResolution:
    nodes: list[GraphNode]
    missing_ids: list[str] = field(default_factory=list)


class SeedResolver:
    """Finds the nodes the explorers start from.

    Seeds come from existing retrievers (vector, hybrid, ...). Their raw results
    must carry the node ``elementId`` in a field called ``elementId``, which
    ``VectorRetriever`` and ``HybridRetriever`` do. Ids that are not found in the
    database of the explorers (for instance because a seed retriever reads another
    database) are reported in :attr:`SeedResolution.missing_ids`. If a seed is a chunk of
    the lexical graph (as created by ``SimpleKGPipeline``), it is replaced by the
    entities extracted from it, with no model call involved.

    Args:
        seed_retrievers (Sequence[Retriever]): Retrievers used to find the seeds.
        graph (GraphAccess): Access to the graph.
        lexical_graph_config (LexicalGraphConfig): Names of the lexical graph.
    """

    def __init__(
        self,
        seed_retrievers: Sequence[Retriever],
        graph: GraphAccess,
        lexical_graph_config: LexicalGraphConfig,
    ) -> None:
        self.seed_retrievers = list(seed_retrievers)
        self.graph = graph
        self.lexical_graph_config = lexical_graph_config

    def _ids_from(self, retriever: Retriever, query_text: str, top_k: int) -> list[str]:
        kwargs: dict[str, Any] = {"query_text": query_text}
        if "top_k" in inspect.signature(retriever.get_search_results).parameters:
            kwargs["top_k"] = top_k
        ids: list[str] = []
        for record in retriever.get_search_results(**kwargs).records:
            node_id = record.get("elementId")
            if node_id is None:
                raise ExploratoryRetrieverError(
                    f"The records of {type(retriever).__name__} have no 'elementId' "
                    "field: it cannot be used as a seed retriever."
                )
            ids.append(str(node_id))
        return ids

    def resolve(self, query_text: str, config: ExplorationConfig) -> SeedResolution:
        """Return up to ``config.num_seeds`` distinct seed nodes."""
        per_retriever = [
            self._ids_from(r, query_text, config.num_seeds)
            for r in self.seed_retrievers
        ]
        ids = _interleave(per_retriever)
        if not ids:
            return SeedResolution([])
        nodes = self.graph.node_details(ids)
        found = {n.id for n in nodes}
        missing = [i for i in ids if i not in found]
        if missing:
            logger.warning(
                "%d seed ids were not found: do the seed retrievers read the same "
                "database as the ExploratoryRetriever?",
                len(missing),
            )
        chunk_label = self.lexical_graph_config.chunk_node_label
        if config.expand_chunks:
            nodes = self._expand_chunks(nodes, chunk_label, config)
        unique: dict[str, GraphNode] = {}
        for node in nodes:
            unique.setdefault(node.id, node)
        return SeedResolution(list(unique.values())[: config.num_seeds], missing)

    def _expand_chunks(
        self, nodes: list[GraphNode], chunk_label: str, config: ExplorationConfig
    ) -> list[GraphNode]:
        chunk_ids = [n.id for n in nodes if chunk_label in n.labels]
        if not chunk_ids:
            return nodes
        entities = self.graph.chunks_to_entities(
            chunk_ids,
            self.lexical_graph_config.node_to_chunk_relationship_type,
            config.max_entities_per_chunk,
        )
        # Chunks are expanded in turn, so that every chunk contributes seeds.
        order: list[list[str]] = []
        for node in nodes:
            if node.id in entities:
                order.append(entities[node.id])
            else:
                order.append([node.id])
        wanted = _interleave(order)
        details = {n.id: n for n in nodes}
        missing = [i for i in wanted if i not in details]
        details.update({n.id: n for n in self.graph.node_details(missing)})
        return [details[i] for i in wanted if i in details]


def _interleave(lists: Sequence[Sequence[str]]) -> list[str]:
    """Round-robin merge of lists, without duplicates."""
    result: list[str] = []
    seen: set[str] = set()
    for i in range(max((len(x) for x in lists), default=0)):
        for items in lists:
            if i < len(items) and items[i] not in seen:
                seen.add(items[i])
                result.append(items[i])
    return result
