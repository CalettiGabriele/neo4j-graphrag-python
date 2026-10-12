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
from typing import Any
from unittest.mock import MagicMock

import pytest
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.experimental.retrievers.exploratory.slots import (
    SlotExtractionTemplate,
    SlotGenerator,
)
from neo4j_graphrag.llm import LLMInterface, LLMInterfaceV2, LLMResponse


def llm_returning(content: str) -> MagicMock:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.return_value = LLMResponse(content=content)
    return llm


def test_prompt_contains_question_and_limit() -> None:
    llm = llm_returning('{"slots": [{"name": "a", "description": "b"}]}')
    SlotGenerator(llm, max_slots=3).generate("Who owns ACME?")
    prompt = llm.invoke.call_args.args[0]
    assert "Who owns ACME?" in prompt and "between 1 and 3" in prompt
    assert llm.invoke.call_args.kwargs["system_instruction"]


def test_parses_slots() -> None:
    llm = llm_returning(
        '{"slots": [{"name": "customer", "description": "Main customer"},'
        '{"name": "suppliers", "description": "Its suppliers"}]}'
    )
    result = SlotGenerator(llm).generate("q")
    assert result.fallback is False
    assert [(s.name, s.description) for s in result.slots] == [
        ("customer", "Main customer"),
        ("suppliers", "Its suppliers"),
    ]


def test_accepts_code_fences_and_bare_lists() -> None:
    fenced = llm_returning(
        '```json\n{"slots": [{"name": "a", "description": "d"}]}\n```'
    )
    assert len(SlotGenerator(fenced).generate("q").slots) == 1
    bare = llm_returning('[{"description": "only a description"}]')
    [slot] = SlotGenerator(bare).generate("q").slots
    assert slot.name == "slot_0"


def test_respects_max_slots() -> None:
    items = ",".join(f'{{"name": "s{i}", "description": "d{i}"}}' for i in range(6))
    result = SlotGenerator(
        llm_returning(f'{{"slots": [{items}]}}'), max_slots=2
    ).generate("q")
    assert [s.name for s in result.slots] == ["s0", "s1"]


@pytest.mark.parametrize(
    "content", ["not json", '{"slots": []}', '{"slots": [{"name": "x"}]}', "42"]
)
def test_unusable_answers_fall_back_to_the_question(content: str) -> None:
    result = SlotGenerator(llm_returning(content)).generate("my question")
    assert result.fallback is True
    [slot] = result.slots
    assert slot.description == "my question"


def test_errors_of_the_llm_call_are_raised() -> None:
    llm = MagicMock(spec=LLMInterface)
    llm.invoke.side_effect = LLMGenerationError("invalid api key")
    with pytest.raises(LLMGenerationError, match="invalid api key"):
        SlotGenerator(llm).generate("my question")


def test_llm_interface_v2_receives_messages() -> None:
    llm = MagicMock(spec=LLMInterfaceV2)
    llm.invoke.return_value = LLMResponse(content='[{"description": "d"}]')
    result = SlotGenerator(llm).generate("q")
    assert result.slots[0].description == "d"
    messages = llm.invoke.call_args.kwargs["input"]
    assert [m["role"] for m in messages] == ["system", "user"]


def test_langchain_chat_model_receives_messages() -> None:
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    llm: Any = FakeListChatModel(responses=['{"slots": [{"description": "d"}]}'])
    result = SlotGenerator(llm).generate("q")
    assert result.fallback is False and result.slots[0].description == "d"


def test_template_requires_its_inputs() -> None:
    assert "{query_text}" in SlotExtractionTemplate.DEFAULT_TEMPLATE
