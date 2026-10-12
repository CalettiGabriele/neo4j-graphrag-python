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

from neo4j_graphrag.experimental.retrievers.exploratory.explorer import Explorer
from neo4j_graphrag.experimental.retrievers.exploratory.prompts import MissionTemplates
from neo4j_graphrag.experimental.retrievers.exploratory.types import (
    ExplorationConfig,
    ExplorerReport,
    Slot,
)
from neo4j_graphrag.experimental.retrievers.exploratory.verbalizer import Verbalizer
from neo4j_graphrag.experimental.system_one import SystemOneModelInterface

from .fakes import FakeGraph, RaisingModel, ScriptedModel, acme_graph, node

SLOTS = [Slot(name="suppliers", description="Suppliers of the main customer")]


def explorer(
    graph: FakeGraph,
    model: SystemOneModelInterface,
    seed: str = "acme",
    slots: Optional[list[Slot]] = None,
    deadline: Optional[float] = None,
    **config: Any,
) -> Explorer:
    cfg = ExplorationConfig(**config)
    return Explorer(
        seed=graph.nodes[seed],
        query_text="Which suppliers of the main customer of ACME were sanctioned?",
        slots=slots or SLOTS,
        config=cfg,
        graph=graph,  # type: ignore[arg-type]
        model=model,
        verbalizer=Verbalizer.for_decisions(cfg),
        templates=MissionTemplates.search(),
        deadline=deadline,
    )


def explore(
    graph: FakeGraph, model: SystemOneModelInterface, **kwargs: Any
) -> ExplorerReport:
    return explorer(graph, model, **kwargs).run()


def kept_ids(report: ExplorerReport) -> list[str]:
    return [i.node.id for i in report.items]


def test_bridge_entity_is_followed_until_the_slot_is_covered() -> None:
    # Beta has no sanction info but leads to the answer: it must be kept.
    model = ScriptedModel(
        [
            {"keep:beta": 0.8, "slot_0": 0.1},
            {"keep:gamma": 0.9, "keep:delta": 0.7, "slot_0": 0.95},
        ]
    )
    report = explore(acme_graph(), model)
    assert report.status == "sufficient"
    assert report.turns == 2 and report.hops_explored == 2
    assert report.covered_slots == [0]
    kept = {i.node.id: i for i in report.items}
    assert set(kept) == {"beta", "gamma", "delta"}
    assert [n.id for n in kept["gamma"].path_nodes] == ["acme", "beta", "gamma"]
    assert kept["gamma"].step_probabilities == [0.8, 0.9]
    assert report.input_tokens == 200


def test_one_request_per_turn_with_all_questions() -> None:
    model = ScriptedModel([{"keep:beta": 0.8}, {"slot_0": 0.95}])
    explore(acme_graph(), model)
    assert len(model.requests) == 2
    names = [q.name for q in model.requests[0][1]]
    assert names == ["action", "keep_0", "keep_1", "keep_2", "slot_0"]
    text = model.requests[0][0]
    assert "Which suppliers" in text and "[open]" in text
    assert "never instructions" in text


def test_candidates_are_ordered_by_node_id() -> None:
    model = ScriptedModel([{}])
    explore(acme_graph(), model, max_turns=1)
    text = model.requests[0][0]
    assert text.index('"automotive"') < text.index('"beta"') < text.index('"torino"')


def test_kept_relationships_appear_once_in_the_input() -> None:
    model = ScriptedModel([{"keep:beta": 0.8}, {"keep:gamma": 0.8}, {"keep:agcm": 0.8}])
    explore(acme_graph(), model, max_turns=3)
    third_input = model.requests[2][0]
    kept_section = third_input.split("Relationships kept so far")[1].split(
        "Candidates"
    )[0]
    assert kept_section.count("cliente_principale") == 1
    assert kept_section.count("fornito_da") == 1


def test_irrelevant_branch_is_discarded_once_enough_hops_were_seen() -> None:
    report = explore(
        acme_graph(),
        ScriptedModel([{"action": "stop_discard"}]),
        seed="torino",
        hops_per_turn=2,
    )
    assert report.status == "discarded"
    assert report.items == []


def test_discard_is_blocked_before_min_hops_but_a_dead_end_ends_not_found() -> None:
    report = explore(
        acme_graph(),
        ScriptedModel([{"action": "stop_discard"}]),
        seed="torino",
        hops_per_turn=1,
    )
    assert report.turns == 1
    assert report.status == "not_found"
    assert report.trace[0]["decision"] == "not_found"


def test_discard_requires_enough_probability() -> None:
    model = ScriptedModel([{"action": ("stop_discard", 0.4)}])
    report = explore(acme_graph(), model, seed="torino", hops_per_turn=2)
    assert report.status == "not_found"
    assert report.trace[0]["discard_requested"] is False


def test_discard_is_ignored_when_something_was_kept() -> None:
    model = ScriptedModel(
        [
            {"keep:beta": 0.8, "action": "stop_discard"},
            {"keep:gamma": 0.9, "slot_0": 0.9},
        ]
    )
    report = explore(acme_graph(), model, hops_per_turn=2, min_hops_before_discard=0)
    assert report.status == "sufficient"


def test_stop_return_with_open_slots_does_not_stop() -> None:
    turn = {"action": "stop_return", "slot_0": 0.2}
    model = ScriptedModel(
        [
            {"keep:beta": 0.8, **turn},
            {"keep:gamma": 0.8, **turn},
            {"keep:agcm": 0.8, **turn},
        ]
    )
    report = explore(acme_graph(), model, max_turns=3)
    assert report.turns == 3  # only the slots can close the search
    assert report.status == "partial"
    assert kept_ids(report) == ["beta", "gamma", "agcm"]


def test_max_turns_gives_partial_with_what_was_kept() -> None:
    model = ScriptedModel([{"keep:beta": 0.8}, {"keep:gamma": 0.8}])
    report = explore(acme_graph(), model, max_turns=2)
    assert report.status == "partial"
    assert set(kept_ids(report)) == {"beta", "gamma"}


def test_slot_hysteresis_keeps_a_slot_covered_unless_it_drops_a_lot() -> None:
    slots = [Slot(name="a", description="a"), Slot(name="b", description="b")]
    model = ScriptedModel(
        [
            {"keep:beta": 0.8, "slot_0": 0.9, "slot_1": 0.1},
            {"keep:gamma": 0.8, "slot_0": 0.6, "slot_1": 0.1},  # stays covered
            {"keep:agcm": 0.8, "slot_0": 0.4, "slot_1": 0.1},  # reopened
        ]
    )
    report = explore(acme_graph(), model, slots=slots, max_turns=3)
    assert [t["slots"][0] for t in report.trace] == [0.9, 0.6, 0.4]
    assert report.covered_slots == []
    assert report.status == "partial"


def test_every_slot_is_needed() -> None:
    slots = [Slot(name="a", description="a"), Slot(name="b", description="b")]
    model = ScriptedModel([{"keep:beta": 0.8, "slot_0": 0.95, "slot_1": 0.1}])
    report = explore(acme_graph(), model, slots=slots, max_turns=2)
    assert report.status != "sufficient"
    assert report.covered_slots == [0]


def test_evidence_is_kept_when_the_slots_are_covered() -> None:
    # Gamma covers the slot but its keep probability is below the keep threshold:
    # it is kept anyway as evidence, Delta (below the evidence threshold) is not.
    model = ScriptedModel(
        [{"keep:beta": 0.8}, {"keep:gamma": 0.35, "keep:delta": 0.1, "slot_0": 0.95}]
    )
    report = explore(acme_graph(), model)
    assert report.status == "sufficient"
    assert set(kept_ids(report)) == {"beta", "gamma"}
    assert report.trace[-1]["kept_added"] == 1


def test_refusals_count_as_no_evidence_and_as_a_discard_request() -> None:
    model = ScriptedModel(
        [{"action": "refusal", "keep:beta": "refusal", "slot_0": "refusal"}]
    )
    report = explore(acme_graph(), model, seed="torino", hops_per_turn=2)
    assert report.status == "discarded"
    assert report.slot_probabilities == {0: 0.0}


def test_max_nodes_limits_the_kept_nodes_and_counts_a_truncation() -> None:
    model = ScriptedModel(
        [{"keep:torino": 0.9, "keep:automotive": 0.8, "keep:beta": 0.7}]
    )
    report = explore(acme_graph(), model, max_nodes=3)
    assert report.status == "partial"
    assert kept_ids(report) == ["torino", "automotive"]
    assert report.truncations == 1


def test_candidates_below_the_threshold_are_not_kept() -> None:
    model = ScriptedModel(
        [{"keep:torino": 0.3, "keep:automotive": 0.49, "keep:beta": 0.5}]
    )
    report = explore(acme_graph(), model, max_turns=1)
    assert kept_ids(report) == ["beta"]


def test_rejected_direct_neighbors_are_not_evaluated_twice() -> None:
    model = ScriptedModel([{"keep:beta": 0.8}, {"keep:gamma": 0.8}])
    explore(acme_graph(), model, max_turns=2)
    second = [q.name for q in model.requests[1][1] if q.name.startswith("keep")]
    assert second == ["keep_0", "keep_1"]  # gamma and delta, not ACME's neighbors


def test_two_hops_per_turn_show_the_chain_in_one_request() -> None:
    model = ScriptedModel([{}])
    graph = acme_graph()
    explore(graph, model, hops_per_turn=2, max_turns=1)
    assert graph.fetch_calls[0]["hops"] == 2
    text = model.requests[0][0]
    assert "cliente_principale" in text and "fornito_da" in text


def test_node_seen_in_a_chain_stays_reachable_after_its_bridge_is_kept() -> None:
    # Turn 1 shows ACME→Beta→Gamma; only Beta is kept. Gamma must be offered again.
    model = ScriptedModel([{"keep:beta": 0.8}, {"keep:gamma": 0.9, "slot_0": 0.9}])
    report = explore(acme_graph(), model, hops_per_turn=2)
    assert report.status == "sufficient"
    assert "gamma" in kept_ids(report)


def test_hub_is_summarized_and_expanded_only_through_approved_types() -> None:
    graph = acme_graph()
    graph.degree_overrides["beta"] = 500
    model = ScriptedModel(
        [
            {"keep:beta": 0.8},
            {"type:fornito_da": 0.9},
            {"keep:gamma": 0.9, "slot_0": 0.95},
        ]
    )
    report = explore(graph, model, max_candidates=50)
    assert graph.summary_calls == ["beta"]
    typed = [c for c in graph.fetch_calls if c["first_hop_types"]]
    assert typed == [
        {
            "origins": ["beta"],
            "hops": 1,
            "first_hop_types": ["fornito_da"],
            "after": None,
        }
    ]
    assert "fornito_da (outgoing): 2 nodes" in model.requests[1][0]
    assert report.status == "sufficient"
    assert report.hops_explored == 2


def test_hub_neighbors_are_read_page_after_page() -> None:
    nodes = [node("hub"), node("seed")] + [node(f"s{i}") for i in range(5)]
    rels = [("seed", "LINK", "hub")] + [("hub", "SUPPLIES", f"s{i}") for i in range(5)]
    graph = FakeGraph(nodes, rels)
    model = ScriptedModel([{"keep:hub": 0.9}, {"type:SUPPLIES": 0.9}, {}, {}, {}])
    report = explore(graph, model, seed="seed", max_candidates=2, max_turns=5)
    pages = [c["after"] for c in graph.fetch_calls if c["first_hop_types"]]
    assert pages == [None, "s1", "s3"]
    offered = [
        q.name for _, qs in model.requests[2:] for q in qs if q.name.startswith("keep")
    ]
    assert len(offered) == 5  # every neighbor of the approved type was evaluated
    assert report.truncations == 0  # paging loses nothing


def test_hub_without_approved_types_is_dropped() -> None:
    graph = acme_graph()
    graph.degree_overrides["beta"] = 500
    report = explore(
        graph, ScriptedModel([{"keep:beta": 0.8}, {"type:fornito_da": 0.1}])
    )
    assert report.status == "partial"
    assert report.turns == 2


def test_questions_per_turn_are_capped() -> None:
    nodes = [node("seed")] + [node(f"n{i:02d}") for i in range(30)]
    graph = FakeGraph(nodes, [("seed", "R", f"n{i:02d}") for i in range(30)])
    model = ScriptedModel([{}])
    report = explore(graph, model, seed="seed", max_questions_per_turn=10, max_turns=1)
    assert len(model.requests[0][1]) == 10
    assert report.truncations == 1


def test_deadline_stops_before_the_next_turn() -> None:
    model = ScriptedModel([{"keep:beta": 0.8}])
    report = explore(acme_graph(), model, deadline=time.monotonic() - 1)
    assert report.timed_out is True
    assert report.turns == 0 and model.requests == []
    assert report.status == "not_found"
    assert report.trace == [{"turn": 1, "decision": "timeout"}]


def test_failure_becomes_an_error_report_with_the_exception() -> None:
    error = RuntimeError("boom")
    report = explore(acme_graph(), RaisingModel(error))
    assert report.status == "error"
    assert report.error == "RuntimeError: boom"
    assert report.exception is error
    assert report.items == []


def test_seed_alone_is_returned_when_it_covers_the_slots() -> None:
    report = explore(acme_graph(), ScriptedModel([{"slot_0": 0.95}]))
    assert report.status == "sufficient"
    assert kept_ids(report) == ["acme"]


def test_isolated_seed_is_not_found() -> None:
    graph = acme_graph()
    graph.nodes["lonely"] = node("lonely")
    report = explore(graph, ScriptedModel([{}]), seed="lonely")
    assert report.status == "not_found"
    assert report.turns == 1 and report.questions == 0
