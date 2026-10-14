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
"""Explore a knowledge graph with the (experimental) ExploratoryRetriever.

Each seed found by the vector search starts an independent explorer. At every
turn an explorer sends one request to a System One Model (here the OpenAI
Decisions API), which answers with probabilities that decide which branches of
the graph to follow and when the information is enough.

Requires: OPENAI_API_KEY, a graph with a vector index on the nodes used as seeds.
"""

import neo4j
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.retrievers import (
    ExplorationConfig,
    ExploratoryRetriever,
)
from neo4j_graphrag.experimental.system_one import OpenAIDecisionModel
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever

# Define database credentials
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "password")
INDEX_NAME = "entity-embeddings"

with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
    # Seeds: the nodes the explorers start from. Any retriever whose raw results
    # contain the node "elementId" can be used (VectorRetriever, HybridRetriever).
    seeds = VectorRetriever(driver, INDEX_NAME, embedder=OpenAIEmbeddings())

    retriever = ExploratoryRetriever(
        driver,
        # Fast model answering closed questions with probabilities.
        system_one_model=OpenAIDecisionModel(),
        # Generates, once per question, the pieces of information to look for
        # ("slots") when a search does not give them. Can be None if every search does.
        slot_llm=OpenAILLM(model_name="gpt-4o"),
        seed_retrievers=[seeds],
        config=ExplorationConfig(
            num_seeds=5,  # one explorer per seed
            max_turns=4,  # at most 4 requests to the System One Model per explorer
            hops_per_turn=1,  # use 2 to let the model see chains of relationships
            max_search_seconds=30,  # no new turn is started after 30 seconds
            # Only these properties are sent to the models ("*": any other label/type)
            included_properties={"*": ["name", "title", "date", "status"]},
        ),
    )

    # Use it on its own...
    result = retriever.search(
        query_text="Which suppliers of the main customer of ACME were sanctioned?"
    )
    for item in result.items:
        print(item.content)
    metadata = result.metadata or {}
    print(metadata.get("status"), metadata.get("slots"))
    print("degraded:", metadata.get("degraded"))

    # Faster: give the slots yourself (e.g. from an agent that planned the question);
    # the LLM call that writes them, often half of the search time, is skipped.
    result = retriever.search(
        query_text="Which suppliers of the main customer of ACME were sanctioned?",
        slots=[
            "the main customer of ACME",
            "the suppliers of that customer",
            "the sanctions received by those suppliers",
        ],
    )
    print((result.metadata or {}).get("slots_source"))  # "caller"

    # ...or in a GraphRAG pipeline. Search options can be overridden per search.
    rag = GraphRAG(retriever=retriever, llm=OpenAILLM(model_name="gpt-4o"))
    answer = rag.search(
        "Which suppliers of the main customer of ACME were sanctioned?",
        retriever_config={"hops_per_turn": 2},
        return_context=True,
    )
    print(answer.answer)
