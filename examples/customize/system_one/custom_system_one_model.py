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
"""Implement a custom System One Model.

A System One Model answers closed questions with probabilities. To use another
provider (for instance a model trained only to classify), subclass
SystemOneModelInterface and implement ``_decide``. Splitting a request when the
provider limits the number of questions, and rate limit handling, are inherited.
"""

from typing import Sequence

from neo4j_graphrag.experimental.system_one import (
    Answer,
    ChoiceAnswer,
    DecisionResponse,
    PredicateAnswer,
    PredicateQuestion,
    Question,
    SystemOneModelInterface,
)


class KeywordDecisionModel(SystemOneModelInterface):
    """Toy model: a predicate is true if the input contains a word of its instructions."""

    # Set it if the provider limits how many questions one request can carry.
    max_questions_per_request = 20

    def __init__(self) -> None:
        super().__init__(model_name="keyword")

    def _decide(self, input: str, questions: Sequence[Question]) -> DecisionResponse:
        answers: list[Answer] = []
        for question in questions:
            if isinstance(question, PredicateQuestion):
                words = {w.strip("?.,").lower() for w in question.instructions.split()}
                hit = any(w in input.lower() for w in words if len(w) > 4)
                answers.append(
                    PredicateAnswer(name=question.name, probability=0.9 if hit else 0.1)
                )
            else:
                # choice and score questions are answered with the first option
                first = getattr(question, "choices", [None])[0]
                if first is not None:
                    answers.append(
                        ChoiceAnswer(
                            name=question.name,
                            choice=first.value,
                            probabilities={first.value: 1.0},
                        )
                    )
        return DecisionResponse(answers=answers)


model = KeywordDecisionModel()
response = model.decide(
    "I was charged twice for my order.",
    [PredicateQuestion(name="billing", instructions="Is this about charged payments?")],
)
print(response.get("billing"))

# It can then be passed to the ExploratoryRetriever:
# ExploratoryRetriever(driver, system_one_model=model, slot_llm=llm, seed_retrievers=[...])
