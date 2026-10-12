import asyncio
from typing import Sequence
from unittest.mock import MagicMock

import pytest
from neo4j_graphrag.exceptions import RateLimitError
from neo4j_graphrag.experimental.system_one import (
    ChoiceAnswer,
    ChoiceOption,
    ChoiceQuestion,
    DecisionResponse,
    PredicateAnswer,
    PredicateQuestion,
    Question,
    RefusalAnswer,
    ScoreLevel,
    ScoreQuestion,
    SystemOneModelError,
    SystemOneModelInterface,
    SystemOneUsage,
)
from neo4j_graphrag.utils.rate_limit import NoOpRateLimitHandler, RetryRateLimitHandler
from pydantic import ValidationError


def predicate(name: str) -> PredicateQuestion:
    return PredicateQuestion(name=name, instructions="?")


class RecordingModel(SystemOneModelInterface):
    def __init__(self, max_questions: int | None = None, **kwargs: object) -> None:
        super().__init__("fake", **kwargs)  # type: ignore[arg-type]
        self.max_questions_per_request = max_questions
        self.requests: list[list[str]] = []

    def _decide(self, input: str, questions: Sequence[Question]) -> DecisionResponse:
        self.requests.append([q.name for q in questions])
        return DecisionResponse(
            answers=[PredicateAnswer(name=q.name, probability=0.5) for q in questions],
            usage=SystemOneUsage(input_tokens=10),
        )


def test_choice_question_needs_two_unique_options() -> None:
    with pytest.raises(ValidationError):
        ChoiceQuestion(
            name="c",
            instructions="?",
            choices=[ChoiceOption(value="a", description="")],
        )
    with pytest.raises(ValidationError):
        ChoiceQuestion(
            name="c",
            instructions="?",
            choices=[
                ChoiceOption(value="a", description=""),
                ChoiceOption(value="a", description=""),
            ],
        )


def test_score_question_needs_two_levels() -> None:
    with pytest.raises(ValidationError):
        ScoreQuestion(
            name="s", instructions="?", levels=[ScoreLevel(label="a", description="")]
        )


def test_probability_must_be_between_0_and_1() -> None:
    with pytest.raises(ValidationError):
        PredicateAnswer(name="p", probability=1.5)


def test_decision_response_get_and_unique_names() -> None:
    response = DecisionResponse(
        answers=[PredicateAnswer(name="a", probability=0.2), RefusalAnswer(name="b")]
    )
    assert isinstance(response.get("a"), PredicateAnswer)
    assert isinstance(response.get("b"), RefusalAnswer)
    assert response.get("missing") is None
    with pytest.raises(ValidationError):
        DecisionResponse(
            answers=[
                PredicateAnswer(name="a", probability=0.2),
                PredicateAnswer(name="a", probability=0.3),
            ]
        )


def test_decide_without_questions_or_with_duplicates_fails() -> None:
    model = RecordingModel()
    with pytest.raises(SystemOneModelError):
        model.decide("x", [])
    with pytest.raises(SystemOneModelError):
        model.decide("x", [predicate("a"), predicate("a")])


def test_decide_single_request_when_no_limit() -> None:
    model = RecordingModel()
    response = model.decide("x", [predicate("a"), predicate("b"), predicate("c")])
    assert model.requests == [["a", "b", "c"]]
    assert [a.name for a in response.answers] == ["a", "b", "c"]
    assert response.usage and response.usage.input_tokens == 10


def test_decide_splits_requests_and_keeps_order() -> None:
    model = RecordingModel(max_questions=2)
    questions = [predicate(n) for n in "abcde"]
    response = model.decide("x", questions)
    assert model.requests == [["a", "b"], ["c", "d"], ["e"]]
    assert [a.name for a in response.answers] == list("abcde")
    assert response.usage and response.usage.input_tokens == 30


def test_decide_reports_missing_answers() -> None:
    class Incomplete(RecordingModel):
        def _decide(
            self, input: str, questions: Sequence[Question]
        ) -> DecisionResponse:
            return DecisionResponse(answers=[PredicateAnswer(name="a", probability=1)])

    with pytest.raises(SystemOneModelError, match="No answer returned"):
        Incomplete().decide("x", [predicate("a"), predicate("b")])


def test_decide_retries_on_rate_limit() -> None:
    calls = MagicMock(side_effect=[RateLimitError("slow down"), None])

    class Flaky(RecordingModel):
        def _decide(
            self, input: str, questions: Sequence[Question]
        ) -> DecisionResponse:
            calls()
            return super()._decide(input, questions)

    handler = RetryRateLimitHandler(
        max_attempts=2, min_wait=0, max_wait=0, jitter=False
    )
    model = Flaky(rate_limit_handler=handler)
    model.decide("x", [predicate("a")])
    assert calls.call_count == 2


def test_decide_does_not_retry_with_noop_handler() -> None:
    class Failing(RecordingModel):
        def _decide(
            self, input: str, questions: Sequence[Question]
        ) -> DecisionResponse:
            raise RateLimitError("slow down")

    with pytest.raises(RateLimitError):
        Failing(rate_limit_handler=NoOpRateLimitHandler()).decide("x", [predicate("a")])


def test_adecide_runs_in_a_thread_and_splits() -> None:
    model = RecordingModel(max_questions=2)
    response = asyncio.run(model.adecide("x", [predicate(n) for n in "abc"]))
    assert sorted(sum(model.requests, [])) == ["a", "b", "c"]
    assert [a.name for a in response.answers] == ["a", "b", "c"]


def test_choice_answer_defaults() -> None:
    answer = ChoiceAnswer(name="c", choice="a")
    assert answer.probabilities == {}
    assert answer.confidence is None


def test_exhausted_retries_raise_the_last_error_not_retry_error() -> None:
    class AlwaysLimited(RecordingModel):
        def _decide(
            self, input: str, questions: Sequence[Question]
        ) -> DecisionResponse:
            raise RateLimitError("slow down")

    handler = RetryRateLimitHandler(
        max_attempts=2, min_wait=0, max_wait=0, jitter=False
    )
    with pytest.raises(RateLimitError, match="slow down"):
        AlwaysLimited(rate_limit_handler=handler).decide("x", [predicate("a")])
    with pytest.raises(RateLimitError, match="slow down"):
        asyncio.run(
            AlwaysLimited(rate_limit_handler=handler).adecide("x", [predicate("a")])
        )
