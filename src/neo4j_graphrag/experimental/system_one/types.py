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
"""Provider-agnostic types for System One Models.

A *System One Model* does not generate free text. It receives an input (the
shared evidence) and a list of closed questions, and it answers each of them
with probabilities. The three question kinds below are the primitives that
providers such as the OpenAI Decisions API expose: ``predicate``, ``choice``
and ``score``.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class PredicateQuestion(BaseModel):
    """A yes/no question. The answer is the probability that the condition is true."""

    type: Literal["predicate"] = "predicate"
    name: str
    instructions: str


class ChoiceOption(BaseModel):
    """One of the values a :class:`ChoiceQuestion` can select."""

    value: str
    description: str


class ChoiceQuestion(BaseModel):
    """A question whose answer is one value among a fixed set of options."""

    type: Literal["choice"] = "choice"
    name: str
    instructions: str
    choices: list[ChoiceOption] = Field(min_length=2)

    @field_validator("choices")
    @classmethod
    def _unique_values(cls, choices: list[ChoiceOption]) -> list[ChoiceOption]:
        values = [c.value for c in choices]
        if len(set(values)) != len(values):
            raise ValueError("choice values must be unique")
        return choices


class ScoreLevel(BaseModel):
    """One ordered level of a :class:`ScoreQuestion`."""

    label: str
    description: str


class ScoreQuestion(BaseModel):
    """A question rated against ordered levels, from the lowest to the highest."""

    type: Literal["score"] = "score"
    name: str
    instructions: str
    levels: list[ScoreLevel] = Field(min_length=2)


Question = Annotated[
    Union[PredicateQuestion, ChoiceQuestion, ScoreQuestion],
    Field(discriminator="type"),
]


class PredicateAnswer(BaseModel):
    type: Literal["predicate"] = "predicate"
    name: str
    probability: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    name: str
    choice: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: Optional[float] = None


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    name: str
    score: float
    probabilities: list[float] = Field(default_factory=list)
    confidence: Optional[float] = None


class RefusalAnswer(BaseModel):
    """The model declined to answer the question."""

    type: Literal["refusal"] = "refusal"
    name: str
    reason: Optional[str] = None


Answer = Annotated[
    Union[PredicateAnswer, ChoiceAnswer, ScoreAnswer, RefusalAnswer],
    Field(discriminator="type"),
]


class SystemOneUsage(BaseModel):
    """Token usage of a request. Decisions-style APIs only bill input tokens."""

    input_tokens: Optional[int] = None


class DecisionResponse(BaseModel):
    """The answers to a list of questions, in the same order as the questions."""

    answers: list[Answer]
    usage: Optional[SystemOneUsage] = None

    @model_validator(mode="after")
    def _unique_names(self) -> "DecisionResponse":
        names = [a.name for a in self.answers]
        if len(set(names)) != len(names):
            raise ValueError("answer names must be unique")
        return self

    def get(
        self, name: str
    ) -> Optional[Union[PredicateAnswer, ChoiceAnswer, ScoreAnswer, RefusalAnswer]]:
        """Return the answer to the question called ``name``, if any."""
        for answer in self.answers:
            if answer.name == name:
                return answer
        return None
