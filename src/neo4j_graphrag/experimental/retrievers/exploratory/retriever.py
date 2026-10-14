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

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import neo4j
from pydantic import ValidationError

from neo4j_graphrag.components.types import LexicalGraphConfig
from neo4j_graphrag.exceptions import (
    RetrieverInitializationError,
    SearchValidationError,
)
from neo4j_graphrag.experimental.system_one import SystemOneModelInterface
from neo4j_graphrag.llm.base import LLMInterface, LLMInterfaceV2
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import (
    LLMModel,
    Neo4jDriverModel,
    RawSearchResult,
    RetrieverResultItem,
)

from .exceptions import ExploratoryRetrieverError
from .explorer import Explorer
from .graph_access import GraphAccess
from .merge import merge_reports
from .prompts import MissionTemplates
from .seeds import SeedResolver
from .slots import SlotGeneration, SlotGenerator
from .types import (
    ExplorationConfig,
    ExploratoryRetrieverModel,
    ExploratorySearchModel,
    ExplorerReport,
    GraphNode,
    Slot,
)
from .verbalizer import NodeFormatter, RelationshipFormatter, Verbalizer

if TYPE_CHECKING:
    from neo4j_graphrag.tool import ObjectParameter, Tool

logger = logging.getLogger(__name__)

DEFAULT_PARAMETER_DESCRIPTIONS = {
    "query_text": "The question to answer with information from the graph.",
    "num_seeds": "Optional. Number of starting points of the exploration (1-10 is typical).",
    "max_turns": "Optional. Maximum number of exploration steps from each starting point.",
    "hops_per_turn": "Optional. Relationships looked ahead at every step, between 1 and 3.",
    "top_k": "Optional. Maximum number of nodes in the results.",
    "slots": (
        "Optional. The pieces of information the answer needs, one short sentence each "
        "(e.g. ['the main customer of ACME', 'the suppliers of that customer']). "
        "Giving them makes the search faster."
    ),
}


class ExploratoryRetriever(Retriever):
    """Retrieves context by exploring the graph with independent explorers guided
    by a System One Model.

    **Experimental**: the interface may change without notice.

    How it works:

    1. ``slot_llm`` splits the question into *slots*, the pieces of information
       the context must contain (one LLM call per search).
    2. ``seed_retrievers`` find the starting nodes; one explorer starts from each.
    3. Every explorer navigates the graph. At each turn it sends **one** request
       to the ``system_one_model``, which answers with probabilities: whether to
       continue or stop, which candidates are worth keeping, and which slots are
       satisfied. The thresholds of ``config`` turn those probabilities into
       decisions. Explorers share nothing and run in parallel.
    4. The reports are merged, deduplicated and ranked. The paths that start from
       the same node are written as one tree, so a relationship shared by several
       paths appears once; each result is such a tree.

    Known limits: questions that need a complete list or an aggregation ("all
    the...", "how many...") cannot be answered reliably by pruned navigation;
    use :class:`~neo4j_graphrag.retrievers.Text2CypherRetriever` for those, for
    example through a :class:`~neo4j_graphrag.retrievers.ToolsRetriever`.
    The search is a single pass and information that has been superseded (for
    instance a revoked penalty) is only found if the properties of the nodes tell so.

    Errors: invalid search arguments raise ``SearchValidationError``; errors of the
    slot LLM are raised; a failing explorer is reported in the metadata, but if
    every explorer fails ``ExploratoryRetrieverError`` is raised. The metadata key
    ``degraded`` tells when the search ran in a degraded mode (slot fallback,
    failed or timed out explorers, truncations, seeds not found).

    Privacy: node and relationship properties are sent to the System One Model
    and returned to the final LLM. Use ``ExplorationConfig.included_properties``
    to choose which ones, by label or type.

    Example:

    .. code-block:: python

        import neo4j
        from neo4j_graphrag.experimental.retrievers import ExploratoryRetriever
        from neo4j_graphrag.experimental.system_one import OpenAIDecisionModel
        from neo4j_graphrag.llm import OpenAILLM
        from neo4j_graphrag.retrievers import VectorRetriever

        driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
        seeds = VectorRetriever(driver, "vector-index-name", embedder)
        retriever = ExploratoryRetriever(
            driver,
            system_one_model=OpenAIDecisionModel(),
            slot_llm=OpenAILLM(model_name="gpt-4o"),
            seed_retrievers=[seeds],
        )
        retriever.search(query_text="Which suppliers of our main customer were sanctioned?")

    Args:
        driver (neo4j.Driver): The Neo4j Python driver.
        system_one_model (SystemOneModelInterface): Model that judges the branches.
        slot_llm (Optional[LLMInterface, LLMInterfaceV2 or LangChain chat model]): LLM that
            generates the slots when a search does not give them, once per search. Can be
            None: every search must then pass its ``slots``.
        seed_retrievers (Sequence[Retriever]): Retrievers that find the seeds. Their raw
            results must contain the ``elementId`` of the nodes.
        config (Optional[ExplorationConfig]): Budget and thresholds.
        lexical_graph_config (Optional[LexicalGraphConfig]): Names used in the lexical graph
            created by ``SimpleKGPipeline``. Chunks found as seeds are replaced by their entities.
        node_formatter (Optional[Callable[[GraphNode], str]]): Custom text of a node.
        relationship_formatter (Optional[Callable[[GraphRelationship], str]]): Custom text of a relationship.
        templates (Optional[MissionTemplates]): Wording of the questions asked to the model.
        max_workers (int): Maximum number of explorers running at the same time.
        return_trace (bool): Include in the metadata of every result the decisions taken by
            the explorers at every turn. Useful to debug and to calibrate the thresholds.
        neo4j_database (Optional[str]): Name of the Neo4j database.

    Raises:
        RetrieverInitializationError: If validation of the input arguments fail.
    """

    # The Neo4j version requirements are checked by the seed retrievers.
    VERIFY_NEO4J_VERSION = False

    def __init__(
        self,
        driver: neo4j.Driver,
        system_one_model: SystemOneModelInterface,
        slot_llm: Optional[Union[LLMInterface, LLMInterfaceV2, Any]],
        seed_retrievers: Sequence[Retriever],
        config: Optional[ExplorationConfig] = None,
        lexical_graph_config: Optional[LexicalGraphConfig] = None,
        node_formatter: Optional[NodeFormatter] = None,
        relationship_formatter: Optional[RelationshipFormatter] = None,
        templates: Optional[MissionTemplates] = None,
        max_workers: int = 8,
        return_trace: bool = False,
        neo4j_database: Optional[str] = None,
    ) -> None:
        try:
            validated = ExploratoryRetrieverModel(
                driver_model=Neo4jDriverModel(driver=driver),
                llm_model=LLMModel(llm=slot_llm) if slot_llm is not None else None,
                system_one_model=system_one_model,
                seed_retrievers=list(seed_retrievers),
                max_workers=max_workers,
                neo4j_database=neo4j_database,
            )
        except ValidationError as e:
            raise RetrieverInitializationError(e.errors()) from e

        super().__init__(validated.driver_model.driver, validated.neo4j_database)
        self.system_one_model = system_one_model
        self.slot_llm = validated.llm_model.llm if validated.llm_model else None
        self.seed_retrievers: list[Retriever] = validated.seed_retrievers
        self.config = config or ExplorationConfig()
        self.lexical_graph_config = lexical_graph_config or LexicalGraphConfig()
        self.templates = templates or MissionTemplates.search()
        self.max_workers = validated.max_workers
        self.return_trace = return_trace
        self._node_formatter = node_formatter
        self._relationship_formatter = relationship_formatter

        lexical = self.lexical_graph_config
        self.graph = GraphAccess(
            self.driver,
            neo4j_database,
            excluded_labels=[
                *self.config.excluded_labels,
                lexical.document_node_label,
            ],
            excluded_relationship_types=[
                *self.config.excluded_relationship_types,
                lexical.next_chunk_relationship_type,
                lexical.chunk_to_document_relationship_type,
            ],
            excluded_properties=self.config.excluded_properties,
            included_properties=self.config.included_properties,
        )
        self.seed_resolver = SeedResolver(
            self.seed_retrievers, self.graph, self.lexical_graph_config
        )

    def default_record_formatter(self, record: neo4j.Record) -> RetrieverResultItem:
        return RetrieverResultItem(
            content=record["content"], metadata=record.get("metadata")
        )

    def convert_to_tool(
        self,
        name: str,
        description: str,
        parameter_descriptions: Optional[dict[str, str]] = None,
    ) -> "Tool":
        """Convert this retriever to a Tool, with default descriptions of its parameters."""
        descriptions = {
            **DEFAULT_PARAMETER_DESCRIPTIONS,
            **(parameter_descriptions or {}),
        }
        return super().convert_to_tool(name, description, descriptions)

    def get_parameters(
        self, parameter_descriptions: Optional[dict[str, str]] = None
    ) -> "ObjectParameter":
        """The tool parameters, with ``slots`` typed as an array of strings (the generic
        inference of the base class only knows arrays of numbers)."""
        from neo4j_graphrag.tool import ArrayParameter, StringParameter

        parameters = super().get_parameters(parameter_descriptions)
        if "slots" in parameters.properties:
            parameters.properties["slots"] = ArrayParameter(
                description=parameters.properties["slots"].description,
                items=StringParameter(description="One piece of information to find."),
                min_items=1,
                max_items=self.config.max_slots,
                required=False,
            )
        return parameters

    def get_search_results(
        self,
        query_text: str,
        num_seeds: Optional[int] = None,
        max_turns: Optional[int] = None,
        hops_per_turn: Optional[int] = None,
        top_k: Optional[int] = None,
        slots: Optional[list[str]] = None,
    ) -> RawSearchResult:
        """Explore the graph and return the paths that are relevant to the question.

        Args:
            query_text (str): The question.
            num_seeds (Optional[int]): Number of seeds, hence of explorers. Overrides the configuration.
            max_turns (Optional[int]): Maximum number of turns of every explorer. Overrides the configuration.
            hops_per_turn (Optional[int]): How many relationships ahead the model sees at every turn
                (1 to 3). Overrides the configuration.
            top_k (Optional[int]): Maximum number of nodes returned. Overrides ``max_output_items``.
                The nodes are presented in groups, see ``ExplorationConfig.group_paths``.
            slots (Optional[list[str]]): The pieces of information the answer needs, one short
                sentence each (at most ``max_slots``). When given, no LLM call is made to
                generate them, which saves the slowest step of a search; an agent that
                already planned the question can pass them. When omitted, ``slot_llm``
                generates them.

        Raises:
            SearchValidationError: If the arguments are not valid.
            ExploratoryRetrieverError: If every explorer failed, for instance because the
                System One Model cannot be reached.

        Returns:
            RawSearchResult: One record per result. With ``group_paths`` (the default) a
            result is a tree: the paths that start from the same node, written once. The
            ``metadata`` of a result tell its score and, for every node it contains, the score and the
            explorers that found it. The metadata of the search describe the slots and the
            outcome of every explorer.
        """
        try:
            search = ExploratorySearchModel.model_validate(
                {
                    "query_text": query_text,
                    "num_seeds": num_seeds,
                    "max_turns": max_turns,
                    "hops_per_turn": hops_per_turn,
                    "top_k": top_k,
                    "slots": slots,
                },
                context={
                    "max_slots": self.config.max_slots,
                    "has_slot_llm": self.slot_llm is not None,
                },
            )
            config = self._config_for(
                num_seeds=search.num_seeds,
                max_turns=search.max_turns,
                hops_per_turn=search.hops_per_turn,
                max_output_items=search.top_k,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors()) from e
        query_text = search.query_text
        deadline = (
            time.monotonic() + config.max_search_seconds
            if config.max_search_seconds is not None
            else None
        )

        if search.slots is not None:
            # Slots given by the caller: no LLM call, the seeds are the only step left.
            generation = SlotGeneration(
                [
                    Slot(name=f"slot_{i}", description=d)
                    for i, d in enumerate(search.slots)
                ]
            )
            slots_source = "caller"
            seeds = self.seed_resolver.resolve(query_text, config)
        else:
            # Slots and seeds do not depend on each other: they are computed together.
            generator = SlotGenerator(self.slot_llm, max_slots=config.max_slots)
            with ThreadPoolExecutor(max_workers=2) as pool:
                slots_future = pool.submit(generator.generate, query_text)
                seeds_future = pool.submit(
                    self.seed_resolver.resolve, query_text, config
                )
                generation = slots_future.result()
                seeds = seeds_future.result()
            slots_source = "generated"
        slot_list = generation.slots
        slot_info = [{"name": s.name, "description": s.description} for s in slot_list]
        degraded: dict[str, Any] = {
            "slots_fallback": generation.fallback,
            "missing_seed_ids": seeds.missing_ids,
        }
        if not seeds.nodes:
            return RawSearchResult(
                records=[],
                metadata={
                    "status": "not_found",
                    "reason": "no_seeds",
                    "slots": slot_info,
                    "slots_source": slots_source,
                    "degraded": degraded,
                },
            )

        decisions = Verbalizer.for_decisions(
            config, self._node_formatter, self._relationship_formatter
        )
        output = Verbalizer.for_output(
            config, self._node_formatter, self._relationship_formatter
        )
        reports = self._explore(
            query_text, seeds.nodes, slot_list, config, decisions, deadline
        )
        self._check_failures(reports)
        merged = merge_reports(reports, slot_list, config, output)

        records = [
            neo4j.Record(
                {
                    "content": group.content,
                    "metadata": {
                        "score": group.score,
                        "explorers": group.explorers,
                        "nodes": [
                            {
                                "node_id": item.node_id,
                                "score": item.score,
                                "explorers": item.explorers,
                                "path_probabilities": item.path_probabilities,
                            }
                            for item in group.items
                        ],
                    },
                }
            )
            for group in merged.groups
        ]
        degraded.update(
            {
                "failed_explorers": sum(r.status == "error" for r in reports),
                "timed_out_explorers": sum(r.timed_out for r in reports),
                "truncations": sum(r.truncations for r in reports),
            }
        )
        metadata: dict[str, Any] = {
            "status": merged.status,
            "slots": [
                {
                    **info,
                    "covered": k in merged.covered_slots,
                    "probability": merged.slot_probabilities.get(k),
                }
                for k, info in enumerate(slot_info)
            ],
            "explorers": [
                {
                    "seed": r.seed_id,
                    "status": r.status,
                    "turns": r.turns,
                    "depth": r.hops_explored,
                    "questions": r.questions,
                    "input_tokens": r.input_tokens,
                    "timed_out": r.timed_out,
                    "truncations": r.truncations,
                    "error": r.error,
                }
                for r in reports
            ],
            "slots_source": slots_source,
            "degraded": degraded,
        }
        if self.return_trace:
            metadata["trace"] = {r.seed_id: r.trace for r in reports}
        return RawSearchResult(records=records, metadata=metadata)

    def _config_for(self, **overrides: Any) -> ExplorationConfig:
        changes = {k: v for k, v in overrides.items() if v is not None}
        if not changes:
            return self.config
        return ExplorationConfig.model_validate({**self.config.model_dump(), **changes})

    @staticmethod
    def _check_failures(reports: Sequence[ExplorerReport]) -> None:
        failed = [r for r in reports if r.status == "error"]
        if not failed:
            return
        logger.warning(
            "%d of %d explorers failed; first error: %s",
            len(failed),
            len(reports),
            failed[0].error,
        )
        if len(failed) == len(reports):
            raise ExploratoryRetrieverError(
                f"All {len(reports)} explorers failed: {failed[0].error}"
            ) from failed[0].exception

    def _explore(
        self,
        query_text: str,
        seeds: Sequence[GraphNode],
        slots: Sequence[Slot],
        config: ExplorationConfig,
        verbalizer: Verbalizer,
        deadline: Optional[float],
    ) -> list[ExplorerReport]:
        def run(seed: GraphNode) -> ExplorerReport:
            return Explorer(
                seed,
                query_text,
                slots,
                config,
                self.graph,
                self.system_one_model,
                verbalizer,
                self.templates,
                deadline=deadline,
            ).run()

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(seeds))) as pool:
            return list(pool.map(run, seeds))
