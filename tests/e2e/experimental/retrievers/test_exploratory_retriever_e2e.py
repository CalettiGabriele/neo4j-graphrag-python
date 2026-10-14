"""End-to-end tests of the ExploratoryRetriever against a real Neo4j database.

The System One Model is replaced by a deterministic, keyword-based one, so the
tests exercise the real Cypher queries and the whole exploration flow without
calling any model provider.
"""

from __future__ import annotations

import re
from typing import Any, Generator, Sequence
from unittest.mock import MagicMock

import neo4j
import pytest
from neo4j_graphrag.experimental.retrievers import (
    ExplorationConfig,
    ExploratoryRetriever,
)
from neo4j_graphrag.experimental.retrievers.exploratory.graph_access import GraphAccess
from neo4j_graphrag.experimental.system_one import (
    ChoiceAnswer,
    DecisionResponse,
    PredicateAnswer,
    Question,
    SystemOneModelInterface,
)
from neo4j_graphrag.llm import LLMInterface, LLMResponse
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RawSearchResult

SLOT_JSON = (
    '{"slots": [{"name": "sanction", '
    '"description": "A sanction received by a supplier of the main customer"}]}'
)


class KeywordModel(SystemOneModelInterface):
    """Keeps the candidates that mention one of the keywords and covers the slot
    as soon as a sanction is visible."""

    def __init__(self, keywords: Sequence[str], slot_keyword: str = "SANCTIONED_BY"):
        super().__init__("keyword")
        self.keywords = keywords
        self.slot_keyword = slot_keyword
        self.requests = 0

    def _decide(self, input: str, questions: Sequence[Question]) -> DecisionResponse:
        self.requests += 1
        candidates = dict(
            (int(i), text)
            for i, text in re.findall(r"^\[(\d+)\] (.*)$", input, flags=re.M)
        )
        groups = dict(
            (int(i), text)
            for i, text in re.findall(r"^\[H(\d+)\] (.*)$", input, flags=re.M)
        )
        answers: list[Any] = []
        for q in questions:
            if q.name == "action":
                answers.append(
                    ChoiceAnswer(
                        name="action",
                        choice="continue",
                        probabilities={"continue": 0.9},
                    )
                )
            elif q.name.startswith("keep_"):
                text = candidates[int(q.name[5:])]
                hit = any(k in text for k in self.keywords)
                answers.append(
                    PredicateAnswer(name=q.name, probability=0.9 if hit else 0.05)
                )
            elif q.name.startswith("type_"):
                text = groups[int(q.name[5:])]
                hit = any(k in text for k in self.keywords)
                answers.append(
                    PredicateAnswer(name=q.name, probability=0.9 if hit else 0.05)
                )
            else:
                covered = self.slot_keyword in input
                answers.append(
                    PredicateAnswer(name=q.name, probability=0.95 if covered else 0.1)
                )
        return DecisionResponse(answers=answers)


class IdsRetriever(Retriever):
    VERIFY_NEO4J_VERSION = False

    def __init__(self, driver: neo4j.Driver, ids: list[str]) -> None:
        super().__init__(driver)
        self.ids = ids

    def get_search_results(self, query_text: str, top_k: int = 5) -> RawSearchResult:
        return RawSearchResult(
            records=[neo4j.Record({"elementId": i}) for i in self.ids[:top_k]]
        )


@pytest.fixture(scope="module")
def ids(driver: neo4j.Driver) -> Generator[dict[str, str], None, None]:
    driver.execute_query("MATCH (n) DETACH DELETE n")
    driver.execute_query(
        """
        CREATE (acme:Company {name: 'ACME', sector: 'Automotive'}),
               (torino:City {name: 'Torino'}),
               (beta:Company {name: 'Beta'}),
               (gamma:Company {name: 'Gamma'}),
               (delta:Company {name: 'Delta'}),
               (agcm:Authority {name: 'AGCM'}),
               (s:Sanction {name: 'Fine 2023', status: 'revoked', year: 2023}),
               (c1:Chunk {text: 'ACME sells to Beta', embedding: range(0, 40)}),
               (c2:Chunk {text: 'Beta is supplied by Gamma'}),
               (doc:Document {path: 'doc.pdf'}),
               (acme)-[:HEADQUARTERED_IN]->(torino),
               (acme)-[:MAIN_CUSTOMER {share: 0.4}]->(beta),
               (beta)-[:SUPPLIED_BY]->(gamma),
               (beta)-[:SUPPLIED_BY]->(delta),
               (gamma)-[:RECEIVED]->(s)-[:SANCTIONED_BY]->(agcm),
               (acme)-[:FROM_CHUNK]->(c1),
               (beta)-[:FROM_CHUNK]->(c1),
               (gamma)-[:FROM_CHUNK]->(c2),
               (c1)-[:NEXT_CHUNK]->(c2),
               (c1)-[:FROM_DOCUMENT]->(doc),
               (c2)-[:FROM_DOCUMENT]->(doc)
        """
    )
    records, _, _ = driver.execute_query(
        "MATCH (n) WHERE n.name IS NOT NULL OR n.text IS NOT NULL "
        "RETURN coalesce(n.name, n.text) AS name, elementId(n) AS id"
    )
    yield {r["name"]: r["id"] for r in records}
    driver.execute_query("MATCH (n) DETACH DELETE n")


def graph_access(driver: neo4j.Driver) -> GraphAccess:
    return GraphAccess(
        driver,
        excluded_labels=["Document"],
        excluded_relationship_types=["NEXT_CHUNK", "FROM_DOCUMENT"],
    )


def build(
    driver: neo4j.Driver,
    seeds: list[str],
    model: SystemOneModelInterface,
    **config: Any,
) -> ExploratoryRetriever:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(content=SLOT_JSON)
    return ExploratoryRetriever(
        driver,
        model,
        llm,
        [IdsRetriever(driver, seeds)],
        config=ExplorationConfig(**config),
    )


def test_graph_access_queries(driver: neo4j.Driver, ids: dict[str, str]) -> None:
    graph = graph_access(driver)

    [node] = graph.node_details([ids["ACME"]])
    assert node.labels == ["Company"] and node.properties["sector"] == "Automotive"

    [chunk] = graph.node_details([ids["ACME sells to Beta"]])
    assert "embedding" not in chunk.properties  # long numeric lists are dropped

    degrees = graph.degrees([ids["ACME"], ids["Beta"]])
    assert degrees[ids["ACME"]] == 3  # city, main customer, chunk
    assert degrees[ids["Beta"]] == 4  # customer, two suppliers, chunk

    one_hop = graph.fetch_paths([ids["ACME"]], 1, [ids["ACME"]], 10)
    assert {
        p.end.properties.get("name") for p in one_hop if "name" in p.end.properties
    } == {
        "Torino",
        "Beta",
    }
    two_hops = graph.fetch_paths([ids["ACME"]], 2, [ids["ACME"]], 50)
    assert any(
        p.end.properties.get("name") == "Gamma" and len(p.relationships) == 2
        for p in two_hops
    )

    typed = graph.fetch_paths(
        [ids["Beta"]], 1, [ids["Beta"]], 10, first_hop_types=["SUPPLIED_BY"]
    )
    assert {p.end.properties["name"] for p in typed} == {"Gamma", "Delta"}

    # Visited nodes are neither returned nor crossed
    blocked = graph.fetch_paths([ids["ACME"]], 2, [ids["ACME"], ids["Beta"]], 50)
    assert all(p.end.properties.get("name") != "Gamma" for p in blocked)

    limited = graph.fetch_paths([ids["Beta"]], 1, [], 1)
    assert len(limited) == 1


def test_lexical_graph_traversal_is_excluded(
    driver: neo4j.Driver, ids: dict[str, str]
) -> None:
    graph = graph_access(driver)
    paths = graph.fetch_paths([ids["ACME sells to Beta"]], 2, [], 50)
    reached = {t for p in paths for r in p.relationships for t in [r.type]}
    assert "NEXT_CHUNK" not in reached and "FROM_DOCUMENT" not in reached


def test_relationship_type_summary(driver: neo4j.Driver, ids: dict[str, str]) -> None:
    summaries = graph_access(driver).relationship_type_summary(ids["Beta"])
    by_key = {(s.type, s.direction): s for s in summaries}
    assert by_key[("SUPPLIED_BY", "out")].count == 2
    assert by_key[("MAIN_CUSTOMER", "in")].count == 1
    assert {n.properties["name"] for n in by_key[("SUPPLIED_BY", "out")].examples} == {
        "Gamma",
        "Delta",
    }


def test_chunks_to_entities(driver: neo4j.Driver, ids: dict[str, str]) -> None:
    result = graph_access(driver).chunks_to_entities(
        [ids["ACME sells to Beta"]], "FROM_CHUNK", 10
    )
    entities = result[ids["ACME sells to Beta"]]
    assert set(entities) == {ids["ACME"], ids["Beta"]}


def test_exploration_follows_the_chain_to_the_sanction(
    driver: neo4j.Driver, ids: dict[str, str]
) -> None:
    model = KeywordModel(["MAIN_CUSTOMER", "SUPPLIED_BY", "RECEIVED", "SANCTIONED_BY"])
    retriever = build(driver, [ids["ACME"]], model, max_turns=5)
    result = retriever.search(
        "Which suppliers of ACME's main customer were sanctioned?"
    )
    assert result.metadata is not None
    assert result.metadata["status"] == "sufficient"
    text = "\n".join(i.content for i in result.items)
    assert "MAIN_CUSTOMER" in text and "SUPPLIED_BY" in text and "SANCTIONED_BY" in text
    # the status of the sanction is part of the context
    assert 'status: "revoked"' in text
    assert "Torino" not in text  # the irrelevant branch was not kept
    # Gamma and Delta share the path from ACME to Beta, which is written once
    assert text.count("MAIN_CUSTOMER") == 1 and text.count("SUPPLIED_BY") == 2
    assert len(result.items) == 1
    assert model.requests <= 5


def test_exploration_with_two_hops_per_turn_needs_fewer_turns(
    driver: neo4j.Driver, ids: dict[str, str]
) -> None:
    keywords = ["MAIN_CUSTOMER", "SUPPLIED_BY", "RECEIVED", "SANCTIONED_BY"]
    one = build(driver, [ids["ACME"]], KeywordModel(keywords), max_turns=6)
    two = build(
        driver, [ids["ACME"]], KeywordModel(keywords), max_turns=6, hops_per_turn=2
    )
    turns_one = one.search("q").metadata["explorers"][0]["turns"]  # type: ignore[index]
    turns_two = two.search("q").metadata["explorers"][0]["turns"]  # type: ignore[index]
    assert turns_two < turns_one


def test_chunk_seeds_start_the_exploration_from_their_entities(
    driver: neo4j.Driver, ids: dict[str, str]
) -> None:
    model = KeywordModel(["SUPPLIED_BY", "RECEIVED", "SANCTIONED_BY"])
    retriever = build(driver, [ids["Beta is supplied by Gamma"]], model, num_seeds=2)
    result = retriever.search("q")
    assert result.metadata is not None
    seeds = {e["seed"] for e in result.metadata["explorers"]}
    assert seeds == {ids["Gamma"]}


def test_hubs_are_explored_through_relationship_types(
    driver: neo4j.Driver, ids: dict[str, str]
) -> None:
    driver.execute_query(
        """
        MATCH (beta:Company {name: 'Beta'})
        UNWIND range(1, 8) AS i
        CREATE (beta)-[:EMPLOYS]->(:Person {name: 'Employee ' + i})
        """
    )
    try:
        model = KeywordModel(["SUPPLIED_BY", "RECEIVED", "SANCTIONED_BY"])
        retriever = build(driver, [ids["Beta"]], model, max_candidates=5, max_turns=5)
        result = retriever.search("q")
        assert result.metadata is not None
        assert result.metadata["status"] == "sufficient"
        text = "\n".join(i.content for i in result.items)
        assert "SUPPLIED_BY" in text and "Employee" not in text
    finally:
        driver.execute_query("MATCH (p:Person) DETACH DELETE p")


def test_unrelated_seed_returns_nothing(
    driver: neo4j.Driver, ids: dict[str, str]
) -> None:
    model = KeywordModel(["NOTHING_MATCHES"])
    result = build(driver, [ids["Torino"]], model).search("q")
    assert result.items == []
    assert result.metadata is not None
    assert result.metadata["status"] == "not_found"


def test_fetch_paths_pages_with_a_cursor(
    driver: neo4j.Driver, ids: dict[str, str]
) -> None:
    graph = graph_access(driver)
    first = graph.fetch_paths([ids["Beta"]], 1, [ids["Beta"]], 2)
    second = graph.fetch_paths(
        [ids["Beta"]], 1, [ids["Beta"]], 2, after=first[-1].end.id
    )
    first_ids = [p.end.id for p in first]
    assert first_ids == sorted(first_ids)
    assert not set(first_ids) & {p.end.id for p in second}
    every = graph.fetch_paths([ids["Beta"]], 1, [ids["Beta"]], 50)
    assert {p.end.id for p in every} == set(first_ids) | {p.end.id for p in second}


def test_degrees_ignore_excluded_and_visited_neighbors(
    driver: neo4j.Driver, ids: dict[str, str]
) -> None:
    graph = graph_access(driver)
    assert graph.degrees([ids["Beta"]], [ids["Gamma"]])[ids["Beta"]] == 3
    chunk = ids["ACME sells to Beta"]
    # NEXT_CHUNK and FROM_DOCUMENT are excluded: only the two FROM_CHUNK count
    assert graph.degrees([chunk])[chunk] == 2


def test_included_properties(driver: neo4j.Driver, ids: dict[str, str]) -> None:
    graph = GraphAccess(driver, included_properties={"Company": ["name"], "*": []})
    [acme, sanction] = graph.node_details([ids["ACME"], ids["Fine 2023"]])
    assert acme.properties == {"name": "ACME"}
    assert sanction.properties == {}
