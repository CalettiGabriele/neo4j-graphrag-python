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
import time
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import neo4j
import pytest
from neo4j_graphrag.exceptions import (
    LLMGenerationError,
    RetrieverInitializationError,
    SearchValidationError,
)
from neo4j_graphrag.experimental.retrievers import (
    ExplorationConfig,
    ExploratoryRetriever,
)
from neo4j_graphrag.experimental.retrievers.exploratory import (
    ExploratoryRetriever as ExploratoryRetrieverFromSubpackage,
)
from neo4j_graphrag.experimental.retrievers.exploratory.exceptions import (
    ExploratoryRetrieverError,
)
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import LLMInterface, LLMInterfaceV2, LLMResponse
from neo4j_graphrag.retrievers import ToolsRetriever
from neo4j_graphrag.retrievers.base import Retriever

from .fakes import FakeGraph, RaisingModel, ScriptedModel, acme_graph
from .stubs import StubRetriever

SLOT_JSON = '{"slots": [{"name": "suppliers", "description": "Suppliers of the main customer"}]}'
DEFAULT_SCRIPT = [{"keep:beta": 0.8}, {"keep:gamma": 0.9, "slot_0": 0.95}]


def slot_llm(content: str = SLOT_JSON) -> MagicMock:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(content=content)
    return llm


def make(
    model: Any = None,
    graph: Optional[FakeGraph] = None,
    seeds: Optional[list[Any]] = None,
    llm: Any = None,
    **kwargs: Any,
) -> ExploratoryRetriever:
    driver = MagicMock(spec=neo4j.Driver)
    seed_retrievers = [StubRetriever(driver, seeds if seeds is not None else ["acme"])]
    retriever = ExploratoryRetriever(
        driver,
        model or ScriptedModel(DEFAULT_SCRIPT),
        llm or slot_llm(),
        seed_retrievers,
        **kwargs,
    )
    fake = graph or acme_graph()
    retriever.graph = fake  # type: ignore[assignment]
    retriever.seed_resolver.graph = fake  # type: ignore[assignment]
    return retriever


def test_search_returns_paths_and_metadata() -> None:
    result = make(return_trace=True).search(
        "Which suppliers of the main customer were sanctioned?"
    )
    assert result.metadata is not None
    assert result.metadata["__retriever"] == "ExploratoryRetriever"
    assert result.metadata["status"] == "sufficient"
    assert result.metadata["slots"][0]["covered"] is True
    [explorer] = result.metadata["explorers"]
    assert explorer["seed"] == "acme" and explorer["status"] == "sufficient"
    assert explorer["turns"] == 2 and explorer["depth"] == 2
    assert explorer["input_tokens"] == 200
    assert result.metadata["degraded"] == {
        "slots_fallback": False,
        "missing_seed_ids": [],
        "failed_explorers": 0,
        "timed_out_explorers": 0,
        "truncations": 0,
    }
    assert result.metadata["trace"]["acme"][0]["turn"] == 1
    # Only maximal paths: ACME→Beta is a prefix of ACME→Beta→Gamma
    contents = [i.content for i in result.items]
    assert len(contents) == 1
    assert "cliente_principale" in contents[0] and "fornito_da" in contents[0]


def test_trace_is_off_by_default() -> None:
    result = make().search("q")
    assert result.metadata is not None and "trace" not in result.metadata


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query_text": ""},
        {"query_text": "   "},
        {"query_text": "q", "hops_per_turn": 7},
        {"query_text": "q", "num_seeds": 0},
        {"query_text": "q", "top_k": -1},
    ],
)
def test_invalid_arguments_raise_search_validation_error(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(SearchValidationError):
        make().search(**kwargs)


def test_top_k_limits_the_results() -> None:
    model = ScriptedModel(
        [{"keep:torino": 0.9, "keep:automotive": 0.9, "keep:beta": 0.9}]
    )
    retriever = make(model=model, config=ExplorationConfig(max_turns=1))
    assert len(retriever.search("q").items) == 3
    assert len(retriever.search("q", top_k=2).items) == 2


def test_graphrag_retriever_config_with_top_k_works() -> None:
    llm = slot_llm("answer")
    rag = GraphRAG(retriever=make(), llm=llm)
    rag.search("q", retriever_config={"top_k": 5}, return_context=False)


def test_no_seeds_gives_not_found_without_running_explorers() -> None:
    model = ScriptedModel([{}])
    result = make(model=model, seeds=[]).search("q")
    assert result.items == []
    assert result.metadata is not None
    assert result.metadata["status"] == "not_found"
    assert result.metadata["reason"] == "no_seeds"
    assert model.requests == []


def test_seed_ids_not_found_are_reported() -> None:
    result = make(seeds=["acme", "ghost"]).search("q")
    assert result.metadata is not None
    assert result.metadata["degraded"]["missing_seed_ids"] == ["ghost"]


def test_explorers_run_one_per_seed() -> None:
    model = ScriptedModel([{"slot_0": 0.95}])
    result = make(model=model, seeds=["acme", "beta", "torino"]).search("q")
    assert result.metadata is not None
    seeds = sorted(e["seed"] for e in result.metadata["explorers"])
    assert seeds == ["acme", "beta", "torino"]


def test_a_failing_explorer_does_not_break_the_search() -> None:
    class Flaky(ScriptedModel):
        def _decide(self, input: str, questions: Any) -> Any:
            if 'first node:\n- Company "beta"' in input:
                raise RuntimeError("boom")
            return super()._decide(input, questions)

    result = make(model=Flaky([{"slot_0": 0.95}]), seeds=["acme", "beta"]).search("q")
    assert result.metadata is not None
    statuses = {e["seed"]: e["status"] for e in result.metadata["explorers"]}
    assert statuses == {"acme": "sufficient", "beta": "error"}
    assert result.metadata["degraded"]["failed_explorers"] == 1
    assert result.items


def test_all_explorers_failing_raises() -> None:
    error = RuntimeError("invalid api key")
    with pytest.raises(
        ExploratoryRetrieverError, match="All 2 explorers failed"
    ) as info:
        make(model=RaisingModel(error), seeds=["acme", "beta"]).search("q")
    assert info.value.__cause__ is error


def test_slot_llm_errors_are_raised() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.side_effect = LLMGenerationError("invalid api key")
    with pytest.raises(LLMGenerationError):
        make(llm=llm).search("q")


def test_slot_fallback_is_reported() -> None:
    result = make(llm=slot_llm("not json")).search("q")
    assert result.metadata is not None
    assert result.metadata["degraded"]["slots_fallback"] is True


def test_slot_llm_can_be_an_llm_interface_v2() -> None:
    llm = MagicMock(spec=LLMInterfaceV2)
    llm.invoke.return_value = LLMResponse(content=SLOT_JSON)
    result = make(llm=llm).search("q")
    assert result.metadata is not None
    assert result.metadata["slots"][0]["name"] == "suppliers"


def test_search_deadline_marks_explorers_as_timed_out() -> None:
    def slow_invoke(*args: Any, **kwargs: Any) -> LLMResponse:
        time.sleep(0.05)
        return LLMResponse(content=SLOT_JSON)

    llm = MagicMock(spec=LLMInterface)
    llm.invoke.side_effect = slow_invoke
    model = ScriptedModel([{}])
    retriever = make(
        model=model, llm=llm, config=ExplorationConfig(max_search_seconds=0.01)
    )
    result = retriever.search("q")
    assert result.metadata is not None
    assert result.metadata["degraded"]["timed_out_explorers"] == 1
    assert result.metadata["explorers"][0]["timed_out"] is True
    assert model.requests == []  # no turn was started after the deadline


def test_parameters_override_the_configuration_for_one_search() -> None:
    model = ScriptedModel([{"keep:beta": 0.8}, {"keep:gamma": 0.8}, {}])
    retriever = make(model=model, config=ExplorationConfig(max_turns=1))
    result = retriever.search("q", max_turns=2)
    assert result.metadata is not None
    assert result.metadata["explorers"][0]["turns"] == 2
    assert retriever.config.max_turns == 1


def test_hops_per_turn_override_reaches_the_graph() -> None:
    graph = acme_graph()
    make(graph=graph, model=ScriptedModel([{}])).search(
        "q", hops_per_turn=2, max_turns=1
    )
    assert graph.fetch_calls[0]["hops"] == 2


def test_graphrag_integration() -> None:
    llm = slot_llm("Gamma")
    retriever = make()
    rag = GraphRAG(retriever=retriever, llm=llm)
    answer = rag.search("Which suppliers?", return_context=True)
    assert answer.answer == "Gamma"
    assert answer.retriever_result is not None and answer.retriever_result.items
    prompt = llm.invoke.call_args.kwargs.get("input") or llm.invoke.call_args.args[0]
    assert "fornito_da" in str(prompt)


def test_can_be_converted_to_a_tool_with_described_parameters() -> None:
    tool = make().convert_to_tool("explore", "Explores the graph")
    parameters = tool.get_parameters()
    params = parameters["properties"]
    assert set(params) == {
        "query_text",
        "num_seeds",
        "max_turns",
        "hops_per_turn",
        "top_k",
    }
    assert params["num_seeds"]["type"] == "integer"
    assert "between 1 and 3" in params["hops_per_turn"]["description"]
    assert parameters["required"] == ["query_text"]
    custom = make().convert_to_tool("e", "d", {"top_k": "How many paths"})
    assert (
        custom.get_parameters()["properties"]["top_k"]["description"]
        == "How many paths"
    )
    assert tool.execute(query_text="q").items


def test_works_inside_a_tools_retriever() -> None:
    tool = make().convert_to_tool("explore", "Explores the graph")
    tools_retriever = ToolsRetriever(
        MagicMock(spec=neo4j.Driver), MagicMock(spec=LLMInterface), [tool]
    )
    assert isinstance(tools_retriever, Retriever)


def test_custom_formatters_are_used() -> None:
    model = ScriptedModel([{"keep_0": 0.9, "keep_1": 0.9, "keep_2": 0.9}])
    retriever = make(
        model=model,
        node_formatter=lambda n: f"<{n.id}>",
        config=ExplorationConfig(max_turns=1),
    )
    contents = [i.content for i in retriever.search("q").items]
    assert contents and all("<" in c for c in contents)


def test_graph_access_gets_lexical_exclusions_and_included_properties() -> None:
    retriever = ExploratoryRetriever(
        MagicMock(spec=neo4j.Driver),
        ScriptedModel([{}]),
        slot_llm(),
        [StubRetriever(MagicMock(spec=neo4j.Driver), [])],
        config=ExplorationConfig(included_properties={"*": ["name"]}),
    )
    assert "NEXT_CHUNK" in retriever.graph.excluded_relationship_types
    assert "FROM_DOCUMENT" in retriever.graph.excluded_relationship_types
    assert "Document" in retriever.graph.excluded_labels
    assert retriever.graph.included_properties == {"*": {"name"}}


def test_package_exports() -> None:
    assert ExploratoryRetrieverFromSubpackage is ExploratoryRetriever


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed_retrievers": []},
        {"system_one_model": object()},
        {"slot_llm": object()},
        {"max_workers": 0},
    ],
)
def test_invalid_constructor_arguments(kwargs: dict[str, Any]) -> None:
    driver = MagicMock(spec=neo4j.Driver)
    args: dict[str, Any] = {
        "driver": driver,
        "system_one_model": ScriptedModel([{}]),
        "slot_llm": slot_llm(),
        "seed_retrievers": [StubRetriever(driver, [])],
    }
    args.update(kwargs)
    with pytest.raises(RetrieverInitializationError):
        ExploratoryRetriever(**args)


def test_does_not_verify_the_neo4j_version() -> None:
    with patch("neo4j_graphrag.retrievers.base.get_version") as get_version:
        make()
    get_version.assert_not_called()
