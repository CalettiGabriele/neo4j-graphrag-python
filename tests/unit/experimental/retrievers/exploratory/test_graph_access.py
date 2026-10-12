from typing import Any
from unittest.mock import MagicMock, patch

import neo4j
import pytest
from neo4j_graphrag.experimental.retrievers.exploratory.exceptions import (
    ExploratoryRetrieverError,
)
from neo4j_graphrag.experimental.retrievers.exploratory.graph_access import GraphAccess

PATCH = (
    "neo4j_graphrag.experimental.retrievers.exploratory.graph_access.get_version_cached"
)


def make(
    version: tuple[int, ...] = (5, 26, 0), **kwargs: object
) -> tuple[GraphAccess, MagicMock]:
    driver = MagicMock(spec=neo4j.Driver)
    driver.execute_query.return_value.records = []
    with patch(PATCH, return_value=(version, False, False)):
        graph = GraphAccess(driver, "db", **kwargs)  # type: ignore[arg-type]
        graph._uses_scoped_subqueries()  # resolve the version while patched
    return graph, driver


def last_call(driver: MagicMock) -> tuple[str, dict[str, Any]]:
    args, kwargs = driver.execute_query.call_args
    assert kwargs["database_"] == "db"
    assert kwargs["routing_"] == neo4j.RoutingControl.READ
    return args[0], args[1]


def test_empty_inputs_do_not_query() -> None:
    graph, driver = make()
    assert graph.node_details([]) == []
    assert graph.degrees([]) == {}
    assert graph.fetch_paths([], 1, [], 10) == []
    assert graph.chunks_to_entities([], "FROM_CHUNK", 5) == {}
    driver.execute_query.assert_not_called()


def test_node_details_preserves_order_and_drops_embeddings() -> None:
    graph, driver = make(excluded_properties=["secret"])
    driver.execute_query.return_value.records = [
        {"id": "b", "labels": ["B"], "properties": {"name": "b"}},
        {
            "id": "a",
            "labels": ["A"],
            "properties": {
                "name": "a",
                "secret": 1,
                "embedding": [0.1] * 50,
                "tags": [1, 2],
            },
        },
    ]
    nodes = graph.node_details(["a", "b", "missing"])
    assert [n.id for n in nodes] == ["a", "b"]
    assert nodes[0].properties == {"name": "a", "tags": [1, 2]}
    query, params = last_call(driver)
    assert "elementId(n) = id" in query and params == {"ids": ["a", "b", "missing"]}


def test_degrees() -> None:
    graph, driver = make(
        excluded_relationship_types=["NEXT_CHUNK"], excluded_labels=["Doc"]
    )
    driver.execute_query.return_value.records = [{"id": "a", "degree": 3}]
    assert graph.degrees(["a"], ["v"]) == {"a": 3}
    query, params = last_call(driver)
    assert (
        "COUNT {" in query and "$excluded_labels" in query and "$exclude_ids" in query
    )
    assert params["excluded_rel_types"] == ["NEXT_CHUNK"]
    assert params["excluded_labels"] == ["Doc"] and params["exclude_ids"] == ["v"]


def test_fetch_paths_query_and_parsing() -> None:
    graph, driver = make(excluded_labels=["Document"])
    driver.execute_query.return_value.records = [
        {
            "origin_id": "a",
            "nodes": [
                {"id": "a", "labels": ["X"], "properties": {"name": "a"}},
                {"id": "b", "labels": ["Y"], "properties": {"name": "b"}},
            ],
            "relationships": [
                {
                    "id": "r",
                    "type": "T",
                    "start_id": "a",
                    "end_id": "b",
                    "properties": {"since": 1},
                }
            ],
        }
    ]
    paths = graph.fetch_paths(["a"], 2, ["z"], 7, first_hop_types=["T"])
    assert len(paths) == 1 and paths[0].end.id == "b"
    assert paths[0].relationships[0].properties == {"since": 1}
    query, params = last_call(driver)
    assert "*1..2" in query and "CALL (start) {" in query
    assert "ORDER BY elementId(end), size(rels) LIMIT $limit" in query
    assert "elementId(end) > $after" in query
    assert params["exclude_ids"] == ["z"] and params["limit"] == 7
    assert params["first_hop_types"] == ["T"] and params["excluded_labels"] == [
        "Document"
    ]


def test_fetch_paths_without_type_restriction_passes_null() -> None:
    graph, driver = make()
    graph.fetch_paths(["a"], 1, [], 5)
    params = last_call(driver)[1]
    assert params["first_hop_types"] is None and params["after"] is None
    graph.fetch_paths(["a"], 1, [], 5, after="id-9")
    assert last_call(driver)[1]["after"] == "id-9"


def test_legacy_subquery_syntax_before_5_23() -> None:
    graph, driver = make(version=(5, 20, 0))
    graph.fetch_paths(["a"], 1, [], 5)
    query = last_call(driver)[0]
    assert "CALL { WITH start MATCH" in query and "CALL (start)" not in query


@pytest.mark.parametrize("hops", [0, 4])
def test_fetch_paths_rejects_invalid_hops(hops: int) -> None:
    graph, _ = make()
    with pytest.raises(ExploratoryRetrieverError):
        graph.fetch_paths(["a"], hops, [], 5)


def test_relationship_type_summary() -> None:
    graph, driver = make()
    driver.execute_query.return_value.records = [
        {
            "type": "SUPPLIES",
            "direction": "in",
            "count": 41,
            "examples": [{"id": "g", "labels": ["C"], "properties": {"name": "Gamma"}}],
        }
    ]
    [summary] = graph.relationship_type_summary(
        "hub", ["x"], max_types=5, num_examples=2
    )
    assert (summary.type, summary.direction, summary.count) == ("SUPPLIES", "in", 41)
    assert summary.examples[0].properties["name"] == "Gamma"
    query, params = last_call(driver)
    assert "COLLECT {" in query and "LIMIT $num_examples" in query
    assert "collect(m)" not in query  # the whole neighborhood is never collected
    assert (
        params["node_id"] == "hub"
        and params["max_types"] == 5
        and params["num_examples"] == 2
    )


def test_chunks_to_entities() -> None:
    graph, driver = make()
    driver.execute_query.return_value.records = [
        {"chunk_id": "c1", "entity_id": "e1"},
        {"chunk_id": "c1", "entity_id": "e2"},
        {"chunk_id": "c2", "entity_id": "e3"},
    ]
    assert graph.chunks_to_entities(["c1", "c2"], "FROM_CHUNK", 3) == {
        "c1": ["e1", "e2"],
        "c2": ["e3"],
    }
    query, params = last_call(driver)
    assert "type(r) = $rel_type" in query and params["rel_type"] == "FROM_CHUNK"
    assert "ORDER BY elementId(e)" in query


def test_version_is_read_once() -> None:
    driver = MagicMock(spec=neo4j.Driver)
    driver.execute_query.return_value.records = []
    graph = GraphAccess(driver)
    with patch(PATCH, return_value=((5, 26, 0), False, False)) as get_version:
        graph.fetch_paths(["a"], 1, [], 5)
        graph.fetch_paths(["a"], 1, [], 5)
    assert get_version.call_count == 1


def test_included_properties_by_label_and_type_with_default() -> None:
    graph, driver = make(
        included_properties={"Person": ["name"], "WORKS_AT": ["since"], "*": ["title"]}
    )
    driver.execute_query.return_value.records = [
        {
            "origin_id": "a",
            "nodes": [
                {
                    "id": "a",
                    "labels": ["Person"],
                    "properties": {"name": "A", "ssn": "x"},
                },
                {
                    "id": "b",
                    "labels": ["Company"],
                    "properties": {"title": "B", "iban": "y"},
                },
            ],
            "relationships": [
                {
                    "id": "r",
                    "type": "WORKS_AT",
                    "start_id": "a",
                    "end_id": "b",
                    "properties": {"since": 2020, "salary": 1},
                }
            ],
        }
    ]
    [path] = graph.fetch_paths(["a"], 1, [], 5)
    assert path.nodes[0].properties == {"name": "A"}
    assert path.nodes[1].properties == {"title": "B"}
    assert path.relationships[0].properties == {"since": 2020}


def test_included_properties_without_default_remove_the_rest() -> None:
    graph, driver = make(included_properties={"Person": ["name"]})
    driver.execute_query.return_value.records = [
        {"id": "b", "labels": ["Company"], "properties": {"name": "B"}}
    ]
    [node] = graph.node_details(["b"])
    assert node.properties == {}
