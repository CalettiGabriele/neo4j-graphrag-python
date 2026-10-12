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

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from tenacity import RetryError

from neo4j_graphrag.utils.rate_limit import (
    DEFAULT_RATE_LIMIT_HANDLER,
    RateLimitHandler,
    async_rate_limit_handler,
    rate_limit_handler,
)

from .exceptions import SystemOneModelError
from .types import Answer, DecisionResponse, Question, SystemOneUsage


class SystemOneModelInterface(ABC):
    """Interface of a System One Model: a fast model that answers closed questions.

    Subclasses only implement :meth:`_decide` (and optionally :meth:`_adecide`).
    The public :meth:`decide` takes care of input validation, of splitting the
    questions over several requests when the provider limits how many questions
    a single request can carry, and of rate limit handling.

    Args:
        model_name (str): Name of the model.
        rate_limit_handler (Optional[RateLimitHandler]): Handler for rate limiting.
            Defaults to retry with exponential backoff.
        **kwargs (Any): Extra arguments for the underlying provider.
    """

    #: Maximum number of questions a single request can carry. ``None`` means
    #: that no limit is known.
    max_questions_per_request: Optional[int] = None

    def __init__(
        self,
        model_name: str,
        rate_limit_handler: Optional[RateLimitHandler] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.model_params = kwargs
        self._rate_limit_handler = rate_limit_handler or DEFAULT_RATE_LIMIT_HANDLER

    @abstractmethod
    def _decide(self, input: str, questions: Sequence[Question]) -> DecisionResponse:
        """Send one request to the provider. Called with at most
        ``max_questions_per_request`` questions."""

    async def _adecide(
        self, input: str, questions: Sequence[Question]
    ) -> DecisionResponse:
        """Async version of :meth:`_decide`. By default runs it in a worker thread."""
        return await asyncio.to_thread(self._decide, input, questions)

    @rate_limit_handler
    def _decide_chunk(
        self, input: str, questions: Sequence[Question]
    ) -> DecisionResponse:
        return self._decide(input, questions)

    @async_rate_limit_handler
    async def _adecide_chunk(
        self, input: str, questions: Sequence[Question]
    ) -> DecisionResponse:
        return await self._adecide(input, questions)

    def decide(self, input: str, questions: Sequence[Question]) -> DecisionResponse:
        """Answer ``questions`` about ``input``.

        Args:
            input (str): The shared evidence the questions refer to.
            questions (Sequence[Question]): Questions with unique names.

        Returns:
            DecisionResponse: One answer per question, in the order of the questions.

        Raises:
            SystemOneModelError: If the questions are invalid or the provider fails.
        """
        chunks = self._split(questions)
        try:
            responses = [self._decide_chunk(input, c) for c in chunks]
        except RetryError as e:
            raise _last_error(e) from e
        return self._merge(responses, questions)

    async def adecide(
        self, input: str, questions: Sequence[Question]
    ) -> DecisionResponse:
        """Async version of :meth:`decide`; chunks are sent concurrently."""
        chunks = self._split(questions)
        try:
            responses = await asyncio.gather(
                *(self._adecide_chunk(input, c) for c in chunks)
            )
        except RetryError as e:
            raise _last_error(e) from e
        return self._merge(list(responses), questions)

    def _split(self, questions: Sequence[Question]) -> list[list[Question]]:
        if not questions:
            raise SystemOneModelError("At least one question is required.")
        names = [q.name for q in questions]
        if len(set(names)) != len(names):
            raise SystemOneModelError("Question names must be unique.")
        size = self.max_questions_per_request or len(questions)
        return [list(questions[i : i + size]) for i in range(0, len(questions), size)]

    @staticmethod
    def _merge(
        responses: list[DecisionResponse], questions: Sequence[Question]
    ) -> DecisionResponse:
        by_name: dict[str, Answer] = {}
        tokens: Optional[int] = None
        for response in responses:
            for answer in response.answers:
                by_name[answer.name] = answer
            if response.usage and response.usage.input_tokens is not None:
                tokens = (tokens or 0) + response.usage.input_tokens
        missing = [q.name for q in questions if q.name not in by_name]
        if missing:
            raise SystemOneModelError(f"No answer returned for questions: {missing}")
        return DecisionResponse(
            answers=[by_name[q.name] for q in questions],
            usage=SystemOneUsage(input_tokens=tokens),
        )


def _last_error(error: RetryError) -> BaseException:
    """The exception of the last attempt, instead of tenacity's ``RetryError``,
    so that callers see the actual cause (for instance ``RateLimitError``)."""
    last = error.last_attempt.exception() if error.last_attempt else None
    return last if last is not None else error
