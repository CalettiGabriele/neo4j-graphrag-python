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

import inspect
import logging
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from neo4j_graphrag.exceptions import RateLimitError
from neo4j_graphrag.utils.rate_limit import RateLimitHandler

from .base import SystemOneModelInterface
from .exceptions import SystemOneModelError
from .types import (
    Answer,
    ChoiceAnswer,
    DecisionResponse,
    PredicateAnswer,
    Question,
    RefusalAnswer,
    ScoreAnswer,
    SystemOneUsage,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

#: Default timeout of a Decisions request, in seconds. The API answers in
#: about 150 ms; the SDK default (600 s) would let a single stuck request
#: block a whole search.
DEFAULT_TIMEOUT = 10.0


def _is_async_http_client(client: Any) -> bool:
    """Whether ``client`` is an async HTTP client (``httpx`` or ``httpx2``)."""
    return inspect.iscoroutinefunction(getattr(client, "send", None))


def split_http_client(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the kwargs of the sync and of the async client.

    ``http_client`` is given only to the client of the matching kind. Unlike
    :func:`neo4j_graphrag.llm.utils.split_http_client_kwargs`, the kind is told
    by duck typing, so that the clients of ``httpx2`` (used by the OpenAI SDK
    from version 3) are recognized as well as those of ``httpx``.
    """
    kwargs = dict(kwargs)
    http_client = kwargs.pop("http_client", None)
    sync_kwargs, async_kwargs = dict(kwargs), dict(kwargs)
    if http_client is not None:
        if _is_async_http_client(http_client):
            async_kwargs["http_client"] = http_client
        else:
            sync_kwargs["http_client"] = http_client
    return sync_kwargs, async_kwargs


class OpenAIDecisionModel(SystemOneModelInterface):
    """System One Model backed by the OpenAI Decisions API (``POST /v1/decisions``).

    With the OpenAI Python SDK 3.26 or later the official ``client.decisions.create``
    is used. Older SDK versions don't have it: the same request is then sent with
    the generic ``client.post``, so the model works with every version allowed by
    the ``openai`` extra.

    Example:

    .. code-block:: python

        from neo4j_graphrag.experimental.system_one import (
            OpenAIDecisionModel,
            PredicateQuestion,
        )

        model = OpenAIDecisionModel()
        response = model.decide(
            "I was charged twice for my order.",
            [PredicateQuestion(name="billing", instructions="Is this about billing?")],
        )

    Args:
        model_name (str): Name of the model. Defaults to ``gpt-6-luna``.
        rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting.
            Defaults to retry with exponential backoff.
        max_questions_per_request (Optional[int]): If set, requests carrying more
            questions are split.
        safety_identifier (Optional[str]): Opaque identifier of the end user, sent
            with every request (see the OpenAI documentation).
        timeout (float): Timeout of a request, in seconds. Defaults to 10.
        **kwargs (Any): Passed to the ``openai.OpenAI`` and ``openai.AsyncOpenAI``
            clients (``api_key``, ``base_url``, ``http_client``, ...). An
            ``http_client`` is given only to the client of the matching kind
            (sync or async). ``max_retries`` defaults to 0 because retries on rate
            limits are already handled by ``rate_limit_handler``.
    """

    client: "OpenAI"
    async_client: "AsyncOpenAI"

    def __init__(
        self,
        model_name: str = "gpt-6-luna",
        rate_limit_handler: Optional[RateLimitHandler] = None,
        max_questions_per_request: Optional[int] = None,
        safety_identifier: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        **kwargs: Any,
    ):
        try:
            import openai
        except ImportError:
            raise ImportError(
                """Could not import openai Python client.
                Please install it with `pip install "neo4j-graphrag[openai]"`."""
            )
        self.openai = openai
        super().__init__(model_name, rate_limit_handler)
        if max_questions_per_request is not None:
            self.max_questions_per_request = max_questions_per_request
        self.safety_identifier = safety_identifier
        kwargs.setdefault("timeout", timeout)
        kwargs.setdefault("max_retries", 0)
        sync_kwargs, async_kwargs = split_http_client(kwargs)
        self.client = openai.OpenAI(**sync_kwargs)
        self.async_client = openai.AsyncOpenAI(**async_kwargs)

    def _params(self, input: str, questions: Sequence[Question]) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_name,
            "input": input,
            "questions": [q.model_dump(mode="json") for q in questions],
        }
        if self.safety_identifier is not None:
            params["safety_identifier"] = self.safety_identifier
        return params

    def _decide(self, input: str, questions: Sequence[Question]) -> DecisionResponse:
        params = self._params(input, questions)
        try:
            if hasattr(self.client, "decisions"):
                raw: Any = self.client.decisions.create(**params)
            else:
                raw = self.client.post("/decisions", cast_to=object, body=params)
        except self.openai.RateLimitError as e:
            raise RateLimitError(f"Rate limit exceeded: {e}") from e
        except self.openai.OpenAIError as e:
            raise SystemOneModelError(f"Decisions request failed: {e}") from e
        return self.parse_response(raw, questions)

    async def _adecide(
        self, input: str, questions: Sequence[Question]
    ) -> DecisionResponse:
        params = self._params(input, questions)
        try:
            if hasattr(self.async_client, "decisions"):
                raw: Any = await self.async_client.decisions.create(**params)
            else:
                raw = await self.async_client.post(
                    "/decisions", cast_to=object, body=params
                )
        except self.openai.RateLimitError as e:
            raise RateLimitError(f"Rate limit exceeded: {e}") from e
        except self.openai.OpenAIError as e:
            raise SystemOneModelError(f"Decisions request failed: {e}") from e
        return self.parse_response(raw, questions)

    @staticmethod
    def parse_response(
        raw: Any, questions: Optional[Sequence[Question]] = None
    ) -> DecisionResponse:
        """Convert a Decisions response into a :class:`DecisionResponse`.

        Args:
            raw: The ``Decision`` object of the SDK, or the JSON body as a dict.
            questions: The questions of the request. The API returns the answers in
                question order and their ``name`` may be null (a refusal of an
                unnamed question, for instance): missing names are then taken from
                the question at the same position.
        """
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        if not isinstance(raw, dict) or not isinstance(raw.get("answers"), list):
            raise SystemOneModelError(f"Unexpected Decisions response: {raw!r}")
        items = raw["answers"]
        if questions is not None and len(items) != len(questions):
            raise SystemOneModelError(
                f"Expected {len(questions)} answers, got {len(items)}"
            )
        answers: list[Answer] = []
        for i, item in enumerate(items):
            fallback = questions[i].name if questions is not None else None
            try:
                answers.append(_parse_answer(item, fallback))
            except (KeyError, TypeError, ValueError) as e:
                raise SystemOneModelError(
                    f"Invalid answer in Decisions response: {item!r}"
                ) from e
        usage = raw.get("usage")
        tokens = None
        if isinstance(usage, dict):
            tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        return DecisionResponse(
            answers=answers, usage=SystemOneUsage(input_tokens=tokens)
        )


def _value(value: Union[str, bool]) -> str:
    # Choice values are typed by the API (string or boolean); the questions of
    # this library only use strings.
    return value if isinstance(value, str) else str(value).lower()


def _parse_answer(item: dict[str, Any], fallback_name: Optional[str]) -> Answer:
    kind = item["type"]
    name = item.get("name") or fallback_name
    if not name:
        raise ValueError("answer without a name")
    if kind == "predicate":
        return PredicateAnswer(name=name, probability=item["probability"])
    if kind == "choice":
        return ChoiceAnswer(
            name=name,
            choice=_value(item["choice"]),
            probabilities={
                _value(p["value"]): p["probability"]
                for p in item.get("probabilities") or []
            },
            confidence=item.get("confidence"),
        )
    if kind == "score":
        levels = sorted(item.get("probabilities") or [], key=lambda p: p["value"])
        return ScoreAnswer(
            name=name,
            score=item["score"],
            probabilities=[p["probability"] for p in levels],
            confidence=item.get("confidence"),
        )
    if kind == "refusal":
        return RefusalAnswer(name=name, reason=item.get("reason"))
    raise ValueError(f"unknown answer type {kind!r}")
