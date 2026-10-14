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

from typing import Any, Callable, Optional, Sequence

from .types import (
    ExplorationConfig,
    GraphNode,
    GraphRelationship,
    RelationshipTypeSummary,
)

NodeFormatter = Callable[[GraphNode], str]
RelationshipFormatter = Callable[[GraphRelationship], str]

#: Properties used, in this order, as the display name of a node.
_NAME_PROPERTIES = ("name", "title", "label", "id")


class _TreeNode:
    """A node of the tree written by :meth:`Verbalizer.tree`."""

    def __init__(
        self,
        node: GraphNode,
        relationship: Optional[GraphRelationship] = None,
        forward: bool = True,
    ) -> None:
        self.node = node
        self.relationship = relationship
        self.forward = forward
        self.children: dict[tuple[str, str], "_TreeNode"] = {}
        self.best = 0.0


class Verbalizer:
    """Turns nodes, relationships and paths into text.

    Two verbalizers are used by a search: a compact one for the System One Model
    (:meth:`for_decisions`) and a richer one for the context returned to the
    final LLM (:meth:`for_output`). The model only sees what is written here, so
    the text carries readable names and the properties that qualify a fact, such
    as dates and status.

    Values coming from the graph are data, not instructions: whitespace is
    collapsed (a value cannot open a new line of the prompt) and strings are
    quoted, with their own quotes escaped.

    Args:
        config (ExplorationConfig): Names of the temporal properties.
        node_formatter (Optional[Callable[[GraphNode], str]]): Custom text of a node.
        relationship_formatter (Optional[Callable[[GraphRelationship], str]]):
            Custom text of a relationship (without its endpoints).
        max_properties (Optional[int]): Properties shown per element. Defaults to
            ``config.max_properties``.
        max_property_chars (Optional[int]): Characters shown per value. Defaults to
            ``config.max_property_chars``.
    """

    def __init__(
        self,
        config: ExplorationConfig,
        node_formatter: Optional[NodeFormatter] = None,
        relationship_formatter: Optional[RelationshipFormatter] = None,
        max_properties: Optional[int] = None,
        max_property_chars: Optional[int] = None,
    ) -> None:
        self.config = config
        self._node_formatter = node_formatter
        self._relationship_formatter = relationship_formatter
        self.max_properties = max_properties or config.max_properties
        self.max_property_chars = max_property_chars or config.max_property_chars

    @classmethod
    def for_decisions(
        cls,
        config: ExplorationConfig,
        node_formatter: Optional[NodeFormatter] = None,
        relationship_formatter: Optional[RelationshipFormatter] = None,
    ) -> "Verbalizer":
        """Compact text, read by the System One Model at every turn."""
        return cls(config, node_formatter, relationship_formatter)

    @classmethod
    def for_output(
        cls,
        config: ExplorationConfig,
        node_formatter: Optional[NodeFormatter] = None,
        relationship_formatter: Optional[RelationshipFormatter] = None,
    ) -> "Verbalizer":
        """Richer text, returned as context to the final LLM."""
        return cls(
            config,
            node_formatter,
            relationship_formatter,
            max_properties=config.output_max_properties,
            max_property_chars=config.output_max_property_chars,
        )

    # -- values and properties ------------------------------------------------

    def _text(self, value: Any) -> str:
        text = " ".join(str(value).split())
        limit = self.max_property_chars
        return text if len(text) <= limit else text[: limit - 3] + "..."

    def _value(self, value: Any) -> str:
        if isinstance(value, str):
            return '"' + self._text(value).replace('"', '\\"') + '"'
        return self._text(value)

    def _properties(self, properties: dict[str, Any], skip: Sequence[str] = ()) -> str:
        temporal = {n.lower() for n in self.config.temporal_property_names}
        items = [
            (k, v)
            for k, v in properties.items()
            if k not in skip and v is not None and v != ""
        ]
        # Temporal/status properties qualify a fact: they are listed first.
        items.sort(key=lambda kv: kv[0].lower() not in temporal)
        items = items[: self.max_properties]
        return ", ".join(f"{self._text(k)}: {self._value(v)}" for k, v in items)

    # -- nodes and relationships ----------------------------------------------

    def node(self, node: GraphNode) -> str:
        if self._node_formatter is not None:
            return self._node_formatter(node)
        label = ":".join(node.labels) or "Node"
        name_key = next((k for k in _NAME_PROPERTIES if node.properties.get(k)), None)
        if name_key is None:
            name_key = next(
                (k for k, v in node.properties.items() if isinstance(v, str) and v),
                None,
            )
        head = (
            f"{label} {self._value(node.properties[name_key])}" if name_key else label
        )
        rest = self._properties(node.properties, skip=[name_key] if name_key else [])
        return f"{head} ({rest})" if rest else head

    def relationship(self, relationship: GraphRelationship) -> str:
        if self._relationship_formatter is not None:
            return self._relationship_formatter(relationship)
        props = self._properties(relationship.properties)
        return f"{relationship.type} {{{props}}}" if props else relationship.type

    def step(
        self, source: GraphNode, relationship: GraphRelationship, target: GraphNode
    ) -> str:
        """One hop, written in the direction of the relationship."""
        rel = self.relationship(relationship)
        if relationship.start_id == source.id:
            return f"{self.node(source)} —[{rel}]→ {self.node(target)}"
        return f"{self.node(source)} ←[{rel}]— {self.node(target)}"

    def path(
        self, nodes: Sequence[GraphNode], relationships: Sequence[GraphRelationship]
    ) -> str:
        """A path as a chain of hops; a single node if there are no relationships."""
        if not relationships:
            return self.node(nodes[0])
        parts = [
            self.step(nodes[i], rel, nodes[i + 1])
            for i, rel in enumerate(relationships)
        ]
        return "; ".join(parts)

    def tree(
        self,
        paths: Sequence[tuple[Sequence[GraphNode], Sequence[GraphRelationship], float]],
    ) -> str:
        """Paths that start from the same node, written as one indented tree.

        Every relationship appears once, however many paths go through it. The
        first line is the root; each following line is a hop, indented by its
        depth: ``- [TYPE] → node`` for a relationship that goes from the parent
        to the node, ``- [TYPE] ← node`` for one that goes from the node to the
        parent. Branches are sorted by the best score found under them.

        Args:
            paths: ``(nodes, relationships, score)`` of every path. All the paths
                must start from the same node.
        """
        root: Optional[_TreeNode] = None
        for nodes, relationships, score in paths:
            if root is None:
                root = _TreeNode(nodes[0])
            elif nodes[0].id != root.node.id:
                raise ValueError(
                    "all the paths of a tree must start from the same node"
                )
            current = root
            current.best = max(current.best, score)
            for i, relationship in enumerate(relationships):
                child_node = nodes[i + 1]
                key = (relationship.id, child_node.id)
                child = current.children.get(key)
                if child is None:
                    forward = relationship.start_id == nodes[i].id
                    child = _TreeNode(child_node, relationship, forward)
                    current.children[key] = child
                child.best = max(child.best, score)
                current = child
        if root is None:
            return ""
        lines = [self.node(root.node)]
        self._tree_lines(root, 0, lines)
        return "\n".join(lines)

    def _tree_lines(self, parent: "_TreeNode", depth: int, lines: list[str]) -> None:
        for child in sorted(parent.children.values(), key=lambda c: -c.best):
            assert child.relationship is not None
            arrow = "→" if child.forward else "←"
            relationship = self.relationship(child.relationship)
            lines.append(
                f"{'  ' * depth}- [{relationship}] {arrow} {self.node(child.node)}"
            )
            self._tree_lines(child, depth + 1, lines)

    def type_summary(self, summary: RelationshipTypeSummary) -> str:
        arrow = "outgoing" if summary.direction == "out" else "incoming"
        examples = ", ".join(self.node(n) for n in summary.examples)
        text = f"{summary.type} ({arrow}): {summary.count} nodes"
        return f"{text}, e.g. {examples}" if examples else text
