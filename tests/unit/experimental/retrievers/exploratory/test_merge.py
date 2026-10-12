import pytest
from neo4j_graphrag.experimental.retrievers.exploratory.merge import (
    merge_reports,
    path_score,
)
from neo4j_graphrag.experimental.retrievers.exploratory.types import (
    ExplorationConfig,
    ExplorerReport,
    GraphNode,
    KeptItem,
    Slot,
)
from neo4j_graphrag.experimental.retrievers.exploratory.verbalizer import Verbalizer

SLOTS = [Slot(name="a", description="a"), Slot(name="b", description="b")]


def item(node_id: str, *probabilities: float, via: tuple[str, ...] = ()) -> KeptItem:
    n = GraphNode(node_id, ["N"], {"name": node_id})
    path = [GraphNode(v, ["N"], {"name": v}) for v in via] + [n]
    return KeptItem(n, path, [], list(probabilities))


def report(
    seed: str,
    status: str,
    items: list[KeptItem],
    covered: list[int],
    probs: dict[int, float],
) -> ExplorerReport:
    return ExplorerReport(
        seed_id=seed,
        status=status,  # type: ignore[arg-type]
        items=items,
        covered_slots=covered,
        slot_probabilities=probs,
    )


def merge(reports: list[ExplorerReport], **config: object):  # type: ignore[no-untyped-def]
    cfg = ExplorationConfig(**config)  # type: ignore[arg-type]
    return merge_reports(reports, SLOTS, cfg, Verbalizer.for_output(cfg))


def test_path_score_is_a_geometric_mean() -> None:
    assert path_score([]) == 1.0
    assert path_score([0.5, 0.5]) == pytest.approx(0.5)
    assert path_score([1.0, 0.25]) == pytest.approx(0.5)
    assert path_score([0.0]) < 1e-6


def test_duplicates_are_merged_with_the_best_path_and_a_bonus() -> None:
    reports = [
        report("s1", "partial", [item("x", 0.6), item("y", 0.9)], [], {0: 0.2}),
        report("s2", "partial", [item("x", 0.8)], [], {0: 0.4}),
    ]
    result = merge(reports, multi_explorer_bonus=0.1)
    by_id = {m.node_id: m for m in result.items}
    assert by_id["x"].score == pytest.approx(0.9)
    assert by_id["x"].explorers == ["s1", "s2"]
    assert by_id["x"].path_probabilities == [0.8]
    assert by_id["y"].score == pytest.approx(0.9)
    assert result.slot_probabilities == {0: 0.4}


def test_items_are_sorted_and_limited() -> None:
    reports = [
        report("s", "partial", [item("a", 0.5), item("b", 0.9), item("c", 0.7)], [], {})
    ]
    result = merge(reports, max_output_items=2)
    assert [m.node_id for m in result.items] == ["b", "c"]


def test_only_sufficient_and_partial_reports_contribute() -> None:
    reports = [
        report("s1", "discarded", [item("a", 0.9)], [0], {0: 0.9}),
        report("s2", "error", [item("b", 0.9)], [0], {0: 0.9}),
        report("s3", "not_found", [], [], {}),
    ]
    result = merge(reports)
    assert result.items == [] and result.status == "not_found"
    assert result.covered_slots == []


def test_status_is_sufficient_when_the_union_covers_every_slot() -> None:
    reports = [
        report("s1", "partial", [item("a", 0.9)], [0], {0: 0.9}),
        report("s2", "partial", [item("b", 0.9)], [1], {1: 0.95}),
    ]
    result = merge(reports)
    assert result.status == "sufficient"
    assert result.covered_slots == [0, 1]


def test_status_is_partial_when_a_slot_is_missing() -> None:
    result = merge([report("s1", "sufficient", [item("a", 0.9)], [0], {0: 0.9})])
    assert result.status == "partial"


def test_content_is_the_verbalized_path() -> None:
    result = merge([report("s", "partial", [item("a", 0.9)], [], {})])
    assert result.items[0].content == 'N "a"'


def test_only_maximal_paths_are_returned() -> None:
    reports = [
        report(
            "s",
            "partial",
            [
                item("b", 0.9, via=("a",)),
                item("c", 0.8, via=("a", "b")),
                item("d", 0.7),
            ],
            [],
            {},
        )
    ]
    assert {m.node_id for m in merge(reports).items} == {"c", "d"}
    assert {m.node_id for m in merge(reports, only_maximal_paths=False).items} == {
        "b",
        "c",
        "d",
    }
