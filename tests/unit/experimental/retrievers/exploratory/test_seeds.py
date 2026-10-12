from unittest.mock import MagicMock

import neo4j
import pytest
from neo4j_graphrag.components.types import LexicalGraphConfig
from neo4j_graphrag.experimental.retrievers.exploratory.exceptions import (
    ExploratoryRetrieverError,
)
from neo4j_graphrag.experimental.retrievers.exploratory.seeds import SeedResolver
from neo4j_graphrag.experimental.retrievers.exploratory.types import ExplorationConfig

from .fakes import FakeGraph, node
from .stubs import NoTopKRetriever, StubRetriever


def driver() -> MagicMock:
    return MagicMock(spec=neo4j.Driver)


def entity_graph() -> FakeGraph:
    nodes = [node(f"e{i}") for i in range(1, 7)]
    nodes += [node("c1", "Chunk"), node("c2", "Chunk"), node("c3", "Chunk")]
    rels = [
        ("e1", "FROM_CHUNK", "c1"),
        ("e2", "FROM_CHUNK", "c1"),
        ("e3", "FROM_CHUNK", "c1"),
        ("e4", "FROM_CHUNK", "c2"),
        ("e5", "FROM_CHUNK", "c2"),
    ]
    return FakeGraph(nodes, rels)


def resolver(graph: FakeGraph, *retrievers: StubRetriever) -> SeedResolver:
    return SeedResolver(retrievers, graph, LexicalGraphConfig())  # type: ignore[arg-type]


def test_seeds_are_read_from_element_ids_and_top_k_is_passed() -> None:
    retriever = StubRetriever(driver(), ["e1", "e2"])
    seeds = (
        resolver(entity_graph(), retriever)
        .resolve("q", ExplorationConfig(num_seeds=4))
        .nodes
    )
    assert [s.id for s in seeds] == ["e1", "e2"]
    assert retriever.calls == [{"query_text": "q", "top_k": 4}]


def test_only_the_element_id_field_is_used() -> None:
    retriever = StubRetriever(driver(), ["e1"], key="id")
    with pytest.raises(ExploratoryRetrieverError, match="elementId"):
        resolver(entity_graph(), retriever).resolve("q", ExplorationConfig())


def test_ids_not_found_are_reported() -> None:
    retriever = StubRetriever(driver(), ["e1", "other-db-id"])
    resolution = resolver(entity_graph(), retriever).resolve("q", ExplorationConfig())
    assert [n.id for n in resolution.nodes] == ["e1"]
    assert resolution.missing_ids == ["other-db-id"]


def test_retrievers_without_top_k_are_called_with_the_query_only() -> None:
    retriever = NoTopKRetriever(driver(), ["e1"])
    resolver(entity_graph(), retriever).resolve("q", ExplorationConfig()).nodes
    assert retriever.calls == [{"query_text": "q"}]


def test_records_without_an_element_id_are_rejected() -> None:
    retriever = StubRetriever(driver(), ["e1"], key="other")
    with pytest.raises(ExploratoryRetrieverError, match="elementId"):
        resolver(entity_graph(), retriever).resolve("q", ExplorationConfig()).nodes


def test_several_retrievers_are_interleaved_without_duplicates() -> None:
    a = StubRetriever(driver(), ["e1", "e2", "e3"])
    b = StubRetriever(driver(), ["e4", "e1", "e5"])
    seeds = (
        resolver(entity_graph(), a, b)
        .resolve("q", ExplorationConfig(num_seeds=10))
        .nodes
    )
    assert [s.id for s in seeds] == ["e1", "e4", "e2", "e3", "e5"]


def test_num_seeds_limits_the_result() -> None:
    retriever = StubRetriever(driver(), ["e1", "e2", "e3"])
    seeds = (
        resolver(entity_graph(), retriever)
        .resolve("q", ExplorationConfig(num_seeds=2))
        .nodes
    )
    assert len(seeds) == 2


def test_no_seeds() -> None:
    resolution = resolver(entity_graph(), StubRetriever(driver(), [])).resolve(
        "q", ExplorationConfig()
    )
    assert resolution.nodes == [] and resolution.missing_ids == []


def test_chunks_are_replaced_by_their_entities_in_turn() -> None:
    retriever = StubRetriever(driver(), ["c1", "c2"])
    seeds = (
        resolver(entity_graph(), retriever)
        .resolve("q", ExplorationConfig(num_seeds=4))
        .nodes
    )
    # Every chunk contributes before any chunk contributes twice
    assert [s.id for s in seeds] == ["e1", "e4", "e2", "e5"]


def test_chunk_without_entities_stays_a_seed() -> None:
    retriever = StubRetriever(driver(), ["c3", "c1"])
    seeds = (
        resolver(entity_graph(), retriever)
        .resolve("q", ExplorationConfig(num_seeds=3))
        .nodes
    )
    assert [s.id for s in seeds] == ["c3", "e1", "e2"]


def test_max_entities_per_chunk() -> None:
    retriever = StubRetriever(driver(), ["c1"])
    seeds = (
        resolver(entity_graph(), retriever)
        .resolve("q", ExplorationConfig(num_seeds=5, max_entities_per_chunk=2))
        .nodes
    )
    assert [s.id for s in seeds] == ["e1", "e2"]


def test_chunk_expansion_can_be_disabled() -> None:
    retriever = StubRetriever(driver(), ["c1"])
    seeds = (
        resolver(entity_graph(), retriever)
        .resolve("q", ExplorationConfig(expand_chunks=False))
        .nodes
    )
    assert [s.id for s in seeds] == ["c1"]


def test_custom_lexical_graph_names() -> None:
    graph = FakeGraph(
        [node("e1"), node("k1", "Passage")], [("e1", "EXTRACTED_FROM", "k1")]
    )
    lexical = LexicalGraphConfig(
        chunk_node_label="Passage", node_to_chunk_relationship_type="EXTRACTED_FROM"
    )
    retriever = StubRetriever(driver(), ["k1"])
    resolver_ = SeedResolver([retriever], graph, lexical)  # type: ignore[arg-type]
    seeds = resolver_.resolve("q", ExplorationConfig()).nodes
    assert [s.id for s in seeds] == ["e1"]
