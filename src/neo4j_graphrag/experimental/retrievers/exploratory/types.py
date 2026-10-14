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

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

from neo4j_graphrag.experimental.system_one import SystemOneModelInterface
from neo4j_graphrag.types import LLMModel, Neo4jDriverModel

ExplorerStatus = Literal["sufficient", "partial", "not_found", "discarded", "error"]


class Slot(BaseModel):
    """An information requirement the retrieved context must satisfy."""

    name: str
    description: str


class ExplorationConfig(BaseModel):
    """Settings of the exploration. Every field can be overridden per search."""

    # Seeds and budget
    num_seeds: int = Field(default=5, ge=1)
    max_turns: int = Field(default=4, ge=1)
    hops_per_turn: int = Field(default=1, ge=1, le=3)
    max_candidates: int = Field(default=50, ge=1)
    max_candidates_per_turn: int = Field(default=100, ge=1)
    max_nodes: int = Field(default=30, ge=1)
    max_slots: int = Field(default=5, ge=1)
    max_output_items: int = Field(default=50, ge=1)
    max_questions_per_turn: int = Field(default=128, ge=8)
    max_search_seconds: Optional[float] = Field(default=60.0, gt=0)

    # Decisions
    min_hops_before_discard: int = Field(default=2, ge=0)
    keep_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    discard_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    slot_covered_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    slot_reopen_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    multi_explorer_bonus: float = Field(default=0.1, ge=0.0)
    evidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # Seeds from a lexical graph
    expand_chunks: bool = True
    max_entities_per_chunk: int = Field(default=10, ge=1)

    # Graph access and verbalization
    excluded_labels: list[str] = Field(default_factory=list)
    excluded_relationship_types: list[str] = Field(default_factory=list)
    excluded_properties: list[str] = Field(default_factory=list)
    included_properties: Optional[dict[str, list[str]]] = None
    temporal_property_names: list[str] = Field(
        default_factory=lambda: [
            "date",
            "start_date",
            "end_date",
            "valid_from",
            "valid_to",
            "status",
            "year",
            "since",
            "until",
        ]
    )
    max_properties: int = Field(default=8, ge=1)
    max_property_chars: int = Field(default=200, ge=10)

    # Output (the context given to the final LLM)
    output_max_properties: int = Field(default=20, ge=1)
    output_max_property_chars: int = Field(default=2000, ge=10)
    only_maximal_paths: bool = True
    group_paths: bool = True

    @model_validator(mode="after")
    def _check_thresholds(self) -> "ExplorationConfig":
        if self.slot_reopen_threshold > self.slot_covered_threshold:
            raise ValueError(
                "slot_reopen_threshold cannot be greater than slot_covered_threshold"
            )
        if self.evidence_threshold > self.keep_threshold:
            raise ValueError("evidence_threshold cannot be greater than keep_threshold")
        return self


@dataclass
class GraphNode:
    id: str
    labels: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelationship:
    id: str
    type: str
    start_id: str
    end_id: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidatePath:
    """A path starting at a frontier node; ``nodes[-1]`` is the candidate."""

    origin_id: str
    nodes: list[GraphNode]
    relationships: list[GraphRelationship]

    @property
    def end(self) -> GraphNode:
        return self.nodes[-1]


@dataclass
class RelationshipTypeSummary:
    """How many neighbors a hub has through one relationship type and direction."""

    node_id: str
    type: str
    direction: Literal["out", "in"]
    count: int
    examples: list[GraphNode] = field(default_factory=list)


@dataclass
class KeptItem:
    """A node an explorer decided to keep, with the full path from the seed.

    ``segment_nodes`` and ``segment_relationships`` are the last part of the path,
    the one added when the node was kept: showing only the segments keeps every
    relationship once in the input of the model.
    """

    node: GraphNode
    path_nodes: list[GraphNode]
    path_relationships: list[GraphRelationship]
    step_probabilities: list[float] = field(default_factory=list)
    segment_nodes: list[GraphNode] = field(default_factory=list)
    segment_relationships: list[GraphRelationship] = field(default_factory=list)


@dataclass
class ExplorerReport:
    seed_id: str
    status: ExplorerStatus
    items: list[KeptItem] = field(default_factory=list)
    slot_probabilities: dict[int, float] = field(default_factory=dict)
    covered_slots: list[int] = field(default_factory=list)
    turns: int = 0
    hops_explored: int = 0
    questions: int = 0
    input_tokens: int = 0
    timed_out: bool = False
    truncations: int = 0
    error: Optional[str] = None
    exception: Optional[BaseException] = field(default=None, repr=False, compare=False)
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MergedItem:
    """A node returned by the search, with the path that leads to it."""

    node_id: str
    content: str
    score: float
    explorers: list[str]
    path_probabilities: list[float]
    path_nodes: list[GraphNode] = field(default_factory=list)
    path_relationships: list[GraphRelationship] = field(default_factory=list)


@dataclass
class ResultGroup:
    """One result of the search: the paths that start from the same node written
    as a tree (or, without grouping, a single path)."""

    content: str
    score: float
    explorers: list[str]
    items: list[MergedItem]


@dataclass
class MergedResult:
    status: Literal["sufficient", "partial", "not_found"]
    items: list[MergedItem]
    groups: list[ResultGroup]
    slot_probabilities: dict[int, float]
    covered_slots: list[int]


class ExploratoryRetrieverModel(BaseModel):
    """Validation of the arguments of the ExploratoryRetriever."""

    driver_model: Neo4jDriverModel
    llm_model: Optional[LLMModel] = None
    system_one_model: Any
    seed_retrievers: list[Any] = Field(min_length=1)
    max_workers: PositiveInt = 8
    neo4j_database: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("system_one_model")
    @classmethod
    def _check_system_one_model(cls, value: Any) -> Any:
        if not isinstance(value, SystemOneModelInterface):
            raise ValueError("system_one_model must be a SystemOneModelInterface")
        return value


SlotDescription = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ExploratorySearchModel(BaseModel):
    """Validation of the arguments of ``ExploratoryRetriever.get_search_results``.

    Validated with a context: ``max_slots`` (the most slots a search may have) and
    ``has_slot_llm`` (whether slots can be generated when none are given).
    """

    query_text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    num_seeds: Optional[PositiveInt] = None
    max_turns: Optional[PositiveInt] = None
    hops_per_turn: Optional[int] = Field(default=None, ge=1, le=3)
    top_k: Optional[PositiveInt] = None
    slots: Optional[list[SlotDescription]] = Field(default=None, min_length=1)

    @field_validator("slots")
    @classmethod
    def _at_most_max_slots(
        cls, value: Optional[list[str]], info: ValidationInfo
    ) -> Optional[list[str]]:
        limit = (info.context or {}).get("max_slots")
        if value is not None and limit is not None and len(value) > limit:
            raise ValueError(f"at most {limit} slots can be given (max_slots)")
        return value

    @model_validator(mode="after")
    def _slots_or_slot_llm(self, info: ValidationInfo) -> "ExploratorySearchModel":
        if self.slots is None and not (info.context or {}).get("has_slot_llm", True):
            raise ValueError(
                "slots are required: the retriever has no slot_llm to generate them"
            )
        return self
