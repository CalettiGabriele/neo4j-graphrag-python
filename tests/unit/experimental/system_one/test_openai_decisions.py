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
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from neo4j_graphrag.exceptions import RateLimitError
from neo4j_graphrag.experimental.system_one import (
    ChoiceAnswer,
    ChoiceOption,
    ChoiceQuestion,
    OpenAIDecisionModel,
    PredicateAnswer,
    PredicateQuestion,
    Question,
    RefusalAnswer,
    ScoreAnswer,
    ScoreLevel,
    ScoreQuestion,
    SystemOneModelError,
)
from neo4j_graphrag.experimental.system_one.openai_decisions import (
    DEFAULT_TIMEOUT,
    split_http_client,
)
from neo4j_graphrag.utils.rate_limit import NoOpRateLimitHandler

RAW: dict[str, Any] = {
    "answers": [
        {"type": "predicate", "name": "p", "probability": 0.92},
        {
            "type": "choice",
            "name": "c",
            "choice": "billing",
            "probabilities": [
                {"value": "billing", "probability": 0.95},
                {"value": "other", "probability": 0.05},
            ],
            "confidence": 0.93,
        },
        {
            "type": "score",
            "name": "s",
            "score": 1.1,
            "probabilities": [
                {"value": 1, "label": "b", "probability": 0.7},
                {"value": 0, "label": "a", "probability": 0.1},
                {"value": 2, "label": "c", "probability": 0.2},
            ],
            "confidence": 0.55,
        },
        {"type": "refusal", "name": None},
    ],
    "model": "gpt-6-luna",
    "usage": {"input_tokens": 42, "output_tokens": 0, "total_tokens": 42},
}

QUESTIONS: list[Question] = [
    PredicateQuestion(name="p", instructions="?"),
    ChoiceQuestion(
        name="c",
        instructions="?",
        choices=[
            ChoiceOption(value="billing", description="b"),
            ChoiceOption(value="other", description="o"),
        ],
    ),
    ScoreQuestion(
        name="s",
        instructions="?",
        levels=[
            ScoreLevel(label="a", description=""),
            ScoreLevel(label="b", description=""),
        ],
    ),
    PredicateQuestion(name="r", instructions="?"),
]


class TypedDecision:
    """Stands for the ``Decision`` object of the OpenAI SDK (3.26+)."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def model_dump(self) -> dict[str, Any]:
        return self.data


@pytest.fixture
def model() -> OpenAIDecisionModel:
    return OpenAIDecisionModel(
        api_key="test", rate_limit_handler=NoOpRateLimitHandler()
    )


def legacy_client(**post: Any) -> MagicMock:
    """A client of an SDK without the ``decisions`` resource."""
    client = MagicMock(spec=["post"])
    client.post = MagicMock(**post)
    return client


def test_parse_response() -> None:
    response = OpenAIDecisionModel.parse_response(RAW, QUESTIONS)
    p, c, s, r = response.answers
    assert isinstance(p, PredicateAnswer) and p.probability == 0.92
    assert isinstance(c, ChoiceAnswer)
    assert c.choice == "billing" and c.probabilities["other"] == 0.05
    assert c.confidence == 0.93
    assert isinstance(s, ScoreAnswer)
    assert s.score == 1.1 and s.probabilities == [0.1, 0.7, 0.2]
    # The refusal has no name: it is matched to the question at the same position.
    assert isinstance(r, RefusalAnswer) and r.name == "r"
    assert response.usage and response.usage.input_tokens == 42


def test_parse_typed_sdk_object() -> None:
    response = OpenAIDecisionModel.parse_response(TypedDecision(RAW), QUESTIONS)
    assert [a.name for a in response.answers] == ["p", "c", "s", "r"]


def test_parse_boolean_choice_values() -> None:
    raw = {
        "answers": [
            {
                "type": "choice",
                "name": "c",
                "choice": True,
                "probabilities": [{"value": True, "probability": 1.0}],
                "confidence": 1.0,
            }
        ]
    }
    [answer] = OpenAIDecisionModel.parse_response(raw).answers
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice == "true" and answer.probabilities == {"true": 1.0}


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {"answers": "x"},
        {"answers": [{"type": "unknown", "name": "a"}]},
        {"answers": [{"type": "predicate", "name": "a"}]},
        {"answers": [{"type": "refusal", "name": None}]},  # no name and no questions
    ],
)
def test_parse_response_invalid(raw: object) -> None:
    with pytest.raises(SystemOneModelError):
        OpenAIDecisionModel.parse_response(raw)


def test_parse_response_with_a_wrong_number_of_answers() -> None:
    with pytest.raises(SystemOneModelError, match="Expected 1 answers, got 4"):
        OpenAIDecisionModel.parse_response(RAW, QUESTIONS[:1])


def test_client_defaults(model: OpenAIDecisionModel) -> None:
    assert model.model_name == "gpt-6-luna"
    assert model.client.timeout == DEFAULT_TIMEOUT
    assert model.client.max_retries == 0
    assert model.async_client.timeout == DEFAULT_TIMEOUT


def test_client_options_can_be_overridden() -> None:
    m = OpenAIDecisionModel(api_key="x", timeout=3.0, max_retries=4)
    assert m.client.timeout == 3.0 and m.client.max_retries == 4


def test_sync_http_client_goes_to_the_sync_client_only() -> None:
    http_client: Any = httpx.Client()
    m = OpenAIDecisionModel(api_key="x", http_client=http_client)
    assert m.client._client is http_client
    assert m.async_client._client is not http_client


def test_async_http_client_goes_to_the_async_client_only() -> None:
    http_client: Any = httpx.AsyncClient()
    m = OpenAIDecisionModel(api_key="x", http_client=http_client)
    assert m.async_client._client is http_client
    assert m.client._client is not http_client


def test_split_http_client_recognizes_any_client_by_duck_typing() -> None:
    class SyncLike:
        def send(self) -> None: ...

    class AsyncLike:
        async def send(self) -> None: ...

    sync_client, async_client = SyncLike(), AsyncLike()
    s, a = split_http_client({"http_client": sync_client, "api_key": "k"})
    assert s == {"http_client": sync_client, "api_key": "k"} and a == {"api_key": "k"}
    s, a = split_http_client({"http_client": async_client})
    assert s == {} and a == {"http_client": async_client}


def test_decide_uses_the_decisions_resource_when_available(
    model: OpenAIDecisionModel,
) -> None:
    model.client = MagicMock()
    model.client.decisions.create.return_value = TypedDecision(RAW)
    response = model.decide("input text", QUESTIONS)
    kwargs = model.client.decisions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-6-luna" and kwargs["input"] == "input text"
    assert [q["type"] for q in kwargs["questions"]] == [
        "predicate",
        "choice",
        "score",
        "predicate",
    ]
    assert "safety_identifier" not in kwargs
    model.client.post.assert_not_called()
    assert [a.name for a in response.answers] == ["p", "c", "s", "r"]


def test_decide_falls_back_to_post_on_older_sdks(model: OpenAIDecisionModel) -> None:
    model.client = legacy_client(return_value=RAW)
    model.decide("input text", QUESTIONS)
    (path,), kwargs = model.client.post.call_args
    assert path == "/decisions"
    assert kwargs["body"]["questions"][1]["choices"][0] == {
        "value": "billing",
        "description": "b",
    }


def test_safety_identifier_is_sent() -> None:
    m = OpenAIDecisionModel(api_key="x", safety_identifier="user-42")
    m.client = MagicMock()
    m.client.decisions.create.return_value = TypedDecision(RAW)
    m.decide("x", QUESTIONS)
    assert m.client.decisions.create.call_args.kwargs["safety_identifier"] == "user-42"


def test_max_questions_per_request_splits(model: OpenAIDecisionModel) -> None:
    model.max_questions_per_request = 2
    model.client = legacy_client(
        side_effect=[{"answers": RAW["answers"][:2]}, {"answers": RAW["answers"][2:]}]
    )
    response = model.decide("x", QUESTIONS)
    assert model.client.post.call_count == 2
    assert [a.name for a in response.answers] == ["p", "c", "s", "r"]


def test_rate_limit_error_is_converted(model: OpenAIDecisionModel) -> None:
    model.client = legacy_client(
        side_effect=openai.RateLimitError(
            "limit", response=MagicMock(status_code=429), body=None
        )
    )
    with pytest.raises(RateLimitError):
        model.decide("x", QUESTIONS[:1])


def test_other_openai_errors_are_wrapped(model: OpenAIDecisionModel) -> None:
    model.client = legacy_client(side_effect=openai.OpenAIError("boom"))
    with pytest.raises(SystemOneModelError, match="boom"):
        model.decide("x", QUESTIONS[:1])


def test_adecide_with_both_sdk_flavours(model: OpenAIDecisionModel) -> None:
    model.async_client = MagicMock()
    model.async_client.decisions.create = AsyncMock(return_value=TypedDecision(RAW))
    assert len(asyncio.run(model.adecide("x", QUESTIONS)).answers) == 4
    model.async_client = MagicMock(spec=["post"])
    model.async_client.post = AsyncMock(return_value=RAW)
    assert len(asyncio.run(model.adecide("x", QUESTIONS)).answers) == 4


def test_adecide_errors(model: OpenAIDecisionModel) -> None:
    model.async_client = MagicMock(spec=["post"])
    model.async_client.post = AsyncMock(side_effect=openai.OpenAIError("boom"))
    with pytest.raises(SystemOneModelError):
        asyncio.run(model.adecide("x", QUESTIONS[:1]))
    model.async_client.post = AsyncMock(
        side_effect=openai.RateLimitError(
            "limit", response=MagicMock(status_code=429), body=None
        )
    )
    with pytest.raises(RateLimitError):
        asyncio.run(model.adecide("x", QUESTIONS[:1]))


def test_real_sdk_request_through_a_mock_transport() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read()
        return httpx.Response(200, json=RAW)

    m = OpenAIDecisionModel(
        api_key="x", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    response = m.decide("hello", QUESTIONS)
    assert seen["url"].endswith("/decisions")
    assert b'"model":"gpt-6-luna"' in seen["body"].replace(b" ", b"")
    assert [a.name for a in response.answers] == ["p", "c", "s", "r"]


def test_missing_openai_raises_import_error() -> None:
    with patch.dict("sys.modules", {"openai": None}):
        with pytest.raises(ImportError, match="neo4j-graphrag\\[openai\\]"):
            OpenAIDecisionModel()
