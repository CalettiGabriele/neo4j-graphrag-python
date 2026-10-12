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

import threading
from typing import Any, Mapping, Optional, Sequence

import neo4j

from neo4j_graphrag.utils.version_utils import (
    get_version_cached,
    is_version_5_23_or_above,
)

from .exceptions import ExploratoryRetrieverError
from .types import (
    CandidatePath,
    GraphNode,
    GraphRelationship,
    RelationshipTypeSummary,
)

#: Properties holding a list longer than this are assumed to be embeddings.
_MAX_LIST_PROPERTY_LENGTH = 20

#: Key of ``included_properties`` applying to the labels and types not listed.
DEFAULT_KEY = "*"

_NODE_PROJECTION = "{id: elementId(%s), labels: labels(%s), properties: properties(%s)}"

_NOT_EXCLUDED = "NONE(l IN labels(%s) WHERE l IN $excluded_labels)"


class GraphAccess:
    """Read-only Cypher access to the graph used by the explorers.

    Every method opens a read transaction through ``driver.execute_query``, which
    makes the class safe to share between the threads running the explorers.
    Results are ordered by ``elementId``, so that a search is reproducible and
    that large neighborhoods can be read page after page.

    Args:
        driver (neo4j.Driver): The Neo4j Python driver.
        neo4j_database (Optional[str]): Name of the database.
        excluded_labels (Sequence[str]): Nodes with one of these labels are never traversed.
        excluded_relationship_types (Sequence[str]): These relationship types are never traversed.
        excluded_properties (Sequence[str]): Properties removed from the nodes and relationships read.
        included_properties (Optional[Mapping[str, Sequence[str]]]): If set, only these
            properties are kept, by node label or relationship type. The key ``"*"``
            applies to the labels and types not listed; without it, their properties
            are all removed.

    Note:
        Element ids are only guaranteed to identify the same element within one
        transaction. A search reads the graph in several transactions, so nodes
        deleted or recreated while a search runs can make it miss or mismatch them.
    """

    def __init__(
        self,
        driver: neo4j.Driver,
        neo4j_database: Optional[str] = None,
        excluded_labels: Sequence[str] = (),
        excluded_relationship_types: Sequence[str] = (),
        excluded_properties: Sequence[str] = (),
        included_properties: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        self.driver = driver
        self.neo4j_database = neo4j_database
        self.excluded_labels = list(excluded_labels)
        self.excluded_relationship_types = list(excluded_relationship_types)
        self.excluded_properties = set(excluded_properties)
        self.included_properties = (
            {k: set(v) for k, v in included_properties.items()}
            if included_properties is not None
            else None
        )
        self._version_is_5_23_or_above: Optional[bool] = None
        self._version_lock = threading.Lock()

    # -- helpers -----------------------------------------------------------

    def _uses_scoped_subqueries(self) -> bool:
        with self._version_lock:
            if self._version_is_5_23_or_above is None:
                version, _, _ = get_version_cached(self.driver, self.neo4j_database)
                self._version_is_5_23_or_above = is_version_5_23_or_above(version)
            return self._version_is_5_23_or_above

    def _subquery(self, imported: str, body: str) -> str:
        if self._uses_scoped_subqueries():
            return f"CALL ({imported}) {{ {body} }} "
        return f"CALL {{ WITH {imported} {body} }} "

    def _run(self, query: str, params: Mapping[str, Any]) -> list[Any]:
        result = self.driver.execute_query(
            query,
            dict(params),
            database_=self.neo4j_database,
            routing_=neo4j.RoutingControl.READ,
        )
        return list(result.records)

    def _allowed(self, keys: Sequence[str]) -> Optional[set[str]]:
        if self.included_properties is None:
            return None
        allowed: set[str] = set()
        listed = False
        for key in keys:
            if key in self.included_properties:
                allowed |= self.included_properties[key]
                listed = True
        if not listed:
            allowed |= self.included_properties.get(DEFAULT_KEY, set())
        return allowed

    def _clean(
        self, properties: Mapping[str, Any], keys: Sequence[str]
    ) -> dict[str, Any]:
        allowed = self._allowed(keys)
        return {
            k: v
            for k, v in properties.items()
            if (allowed is None or k in allowed)
            and k not in self.excluded_properties
            and not (isinstance(v, list) and len(v) > _MAX_LIST_PROPERTY_LENGTH)
        }

    def _node(self, data: Mapping[str, Any]) -> GraphNode:
        labels = list(data.get("labels") or [])
        return GraphNode(
            id=data["id"],
            labels=labels,
            properties=self._clean(data.get("properties") or {}, labels),
        )

    def _relationship(self, data: Mapping[str, Any]) -> GraphRelationship:
        return GraphRelationship(
            id=data["id"],
            type=data["type"],
            start_id=data["start_id"],
            end_id=data["end_id"],
            properties=self._clean(data.get("properties") or {}, [data["type"]]),
        )

    def _base_params(self) -> dict[str, Any]:
        return {
            "excluded_rel_types": self.excluded_relationship_types,
            "excluded_labels": self.excluded_labels,
        }

    # -- queries -----------------------------------------------------------

    def node_details(self, node_ids: Sequence[str]) -> list[GraphNode]:
        """Read labels and properties of the given nodes, preserving the order.
        Ids that do not exist are skipped."""
        if not node_ids:
            return []
        query = (
            "UNWIND $ids AS id MATCH (n) WHERE elementId(n) = id "
            "RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties"
        )
        by_id = {
            r["id"]: self._node(r) for r in self._run(query, {"ids": list(node_ids)})
        }
        return [by_id[i] for i in node_ids if i in by_id]

    def degrees(
        self, node_ids: Sequence[str], exclude_ids: Sequence[str] = ()
    ) -> dict[str, int]:
        """Number of neighbors of each node that can be traversed: excluded types,
        excluded labels and ``exclude_ids`` are not counted."""
        if not node_ids:
            return {}
        query = (
            "UNWIND $ids AS id MATCH (n) WHERE elementId(n) = id "
            "RETURN id, COUNT { (n)-[r]-(m) WHERE NOT type(r) IN $excluded_rel_types "
            f"AND {_NOT_EXCLUDED % 'm'} "
            "AND NOT elementId(m) IN $exclude_ids } AS degree"
        )
        params = {
            "ids": list(node_ids),
            "exclude_ids": list(exclude_ids),
            **self._base_params(),
        }
        return {r["id"]: int(r["degree"]) for r in self._run(query, params)}

    def fetch_paths(
        self,
        origin_ids: Sequence[str],
        hops: int,
        exclude_ids: Sequence[str],
        limit: int,
        first_hop_types: Optional[Sequence[str]] = None,
        after: Optional[str] = None,
    ) -> list[CandidatePath]:
        """Paths of 1 to ``hops`` relationships starting at each origin node.

        Paths are ordered by the ``elementId`` of their end node, then by length,
        and at most ``limit`` are returned for each origin. ``after`` reads the next
        page: only paths whose end node id is greater than it are returned (keyset
        pagination, which stays correct while ``exclude_ids`` grows between pages). Paths ending in, or passing through, one of ``exclude_ids`` are
        not returned. If ``first_hop_types`` is set, the first relationship of
        every path must have one of those types.
        """
        if not origin_ids:
            return []
        if not 1 <= hops <= 3:
            raise ExploratoryRetrieverError("hops must be between 1 and 3")
        body = (
            f"MATCH p = (start)-[rels*1..{int(hops)}]-(end) "
            "WHERE end <> start AND NOT elementId(end) IN $exclude_ids "
            "AND NONE(m IN nodes(p)[1..-1] WHERE elementId(m) IN $exclude_ids) "
            "AND ALL(r IN rels WHERE NOT type(r) IN $excluded_rel_types) "
            f"AND ALL(n IN nodes(p) WHERE {_NOT_EXCLUDED % 'n'}) "
            "AND ($first_hop_types IS NULL OR type(head(rels)) IN $first_hop_types) "
            "AND ($after IS NULL OR elementId(end) > $after) "
            "RETURN p AS path ORDER BY elementId(end), size(rels) LIMIT $limit"
        )
        query = (
            "UNWIND $origin_ids AS origin_id "
            "MATCH (start) WHERE elementId(start) = origin_id "
            + self._subquery("start", body)
            + "RETURN origin_id, "
            f"[n IN nodes(path) | {_NODE_PROJECTION % ('n', 'n', 'n')}] AS nodes, "
            "[r IN relationships(path) | {id: elementId(r), type: type(r), "
            "start_id: elementId(startNode(r)), end_id: elementId(endNode(r)), "
            "properties: properties(r)}] AS relationships"
        )
        params = {
            "origin_ids": list(origin_ids),
            "exclude_ids": list(exclude_ids),
            "limit": limit,
            "after": after,
            "first_hop_types": list(first_hop_types) if first_hop_types else None,
            **self._base_params(),
        }
        return [
            CandidatePath(
                origin_id=r["origin_id"],
                nodes=[self._node(n) for n in r["nodes"]],
                relationships=[self._relationship(x) for x in r["relationships"]],
            )
            for r in self._run(query, params)
        ]

    def relationship_type_summary(
        self,
        node_id: str,
        exclude_ids: Sequence[str] = (),
        max_types: int = 12,
        num_examples: int = 3,
    ) -> list[RelationshipTypeSummary]:
        """Neighbors of a node grouped by relationship type and direction.

        Only the counts are aggregated over the whole neighborhood: the examples
        are read with a bounded subquery, so that supernodes stay cheap.
        """
        neighbor_filter = "NOT elementId(%s) IN $exclude_ids AND " + _NOT_EXCLUDED
        query = (
            "MATCH (n) WHERE elementId(n) = $node_id "
            "MATCH (n)-[r]-(m) WHERE NOT type(r) IN $excluded_rel_types "
            f"AND {neighbor_filter % ('m', 'm')} "
            "WITH n, type(r) AS type, "
            "CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END AS direction, "
            "count(m) AS count "
            "ORDER BY count DESC, type, direction LIMIT $max_types "
            "RETURN type, direction, count, COLLECT { "
            "MATCH (n)-[r2]-(x) WHERE type(r2) = type "
            "AND (startNode(r2) = n) = (direction = 'out') "
            f"AND {neighbor_filter % ('x', 'x')} "
            f"RETURN {_NODE_PROJECTION % ('x', 'x', 'x')} "
            "ORDER BY elementId(x) LIMIT $num_examples } AS examples"
        )
        params = {
            "node_id": node_id,
            "exclude_ids": list(exclude_ids),
            "max_types": max_types,
            "num_examples": num_examples,
            **self._base_params(),
        }
        return [
            RelationshipTypeSummary(
                node_id=node_id,
                type=r["type"],
                direction=r["direction"],
                count=int(r["count"]),
                examples=[self._node(x) for x in r["examples"]],
            )
            for r in self._run(query, params)
        ]

    def chunks_to_entities(
        self, chunk_ids: Sequence[str], relationship_type: str, limit: int
    ) -> dict[str, list[str]]:
        """Entities extracted from each chunk, following ``(entity)-[type]->(chunk)``."""
        if not chunk_ids:
            return {}
        body = (
            "MATCH (e)-[r]->(c) WHERE type(r) = $rel_type "
            f"AND {_NOT_EXCLUDED % 'e'} "
            "RETURN e ORDER BY elementId(e) LIMIT $limit"
        )
        query = (
            "UNWIND $ids AS chunk_id MATCH (c) WHERE elementId(c) = chunk_id "
            + self._subquery("c", body)
            + "RETURN chunk_id, elementId(e) AS entity_id"
        )
        params = {
            "ids": list(chunk_ids),
            "rel_type": relationship_type,
            "limit": limit,
            **self._base_params(),
        }
        result: dict[str, list[str]] = {}
        for r in self._run(query, params):
            result.setdefault(r["chunk_id"], []).append(r["entity_id"])
        return result
