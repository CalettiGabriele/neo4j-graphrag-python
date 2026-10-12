from typing import Any, Optional

import neo4j
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RawSearchResult


class StubRetriever(Retriever):
    """A seed retriever returning fixed node ids."""

    VERIFY_NEO4J_VERSION = False

    def __init__(
        self, driver: neo4j.Driver, ids: list[Optional[str]], key: str = "elementId"
    ) -> None:
        super().__init__(driver)
        self.ids = ids
        self.key = key
        self.calls: list[dict[str, Any]] = []

    def get_search_results(self, query_text: str, top_k: int = 5) -> RawSearchResult:
        self.calls.append({"query_text": query_text, "top_k": top_k})
        return RawSearchResult(
            records=[neo4j.Record({self.key: i, "score": 1.0}) for i in self.ids]
        )


class NoTopKRetriever(StubRetriever):
    def get_search_results(self, query_text: str) -> RawSearchResult:  # type: ignore[override]
        self.calls.append({"query_text": query_text})
        return RawSearchResult(
            records=[neo4j.Record({"elementId": i}) for i in self.ids]
        )
