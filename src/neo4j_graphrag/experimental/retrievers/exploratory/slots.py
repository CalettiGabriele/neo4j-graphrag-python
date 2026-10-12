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

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Union

from neo4j_graphrag.generation.prompts import PromptTemplate
from neo4j_graphrag.llm.base import LLMInterface, LLMInterfaceV2
from neo4j_graphrag.llm.utils import legacy_inputs_to_messages

from .types import Slot

logger = logging.getLogger(__name__)


class SlotExtractionTemplate(PromptTemplate):
    """Prompt that splits a question into the pieces of information needed to answer it."""

    DEFAULT_SYSTEM_INSTRUCTIONS = (
        "You break user questions down into information requirements."
    )
    DEFAULT_TEMPLATE = """Break the question below down into the distinct pieces of \
information that must be found in a knowledge graph in order to answer it.

Rules:
- Return between 1 and {max_slots} requirements.
- Each requirement must be specific and verifiable on its own.
- If a requirement depends on another one (for example the suppliers of "the main \
customer" depend on knowing who the main customer is), keep them as separate requirements.
- Answer with JSON only, in this format:
{{"slots": [{{"name": "short_snake_case_name", "description": "what must be found"}}]}}

Question: {query_text}

JSON:"""
    EXPECTED_INPUTS = ["query_text", "max_slots"]


@dataclass
class SlotGeneration:
    slots: list[Slot]
    #: True if the answer of the LLM could not be used and the question itself
    #: became the only slot.
    fallback: bool = False


class SlotGenerator:
    """Generates the information requirements (slots) of a question with an LLM.

    The LLM can be an :class:`~neo4j_graphrag.llm.LLMInterface`, an
    :class:`~neo4j_graphrag.llm.LLMInterfaceV2` or a LangChain chat model, as in
    :class:`~neo4j_graphrag.generation.GraphRAG`.

    Errors of the LLM call (authentication, network, rate limits) are raised. If
    the LLM answers but its answer cannot be parsed, a single slot equal to the
    question is used and :attr:`SlotGeneration.fallback` is set.

    Args:
        llm: The LLM used to generate the slots.
        template (SlotExtractionTemplate): The prompt.
        max_slots (int): Maximum number of slots.
    """

    def __init__(
        self,
        llm: Union[LLMInterface, LLMInterfaceV2, Any],
        template: Optional[SlotExtractionTemplate] = None,
        max_slots: int = 5,
    ) -> None:
        self.llm = llm
        self.template = template or SlotExtractionTemplate()
        self.max_slots = max_slots

    def _invoke(self, prompt: str) -> str:
        system = self.template.system_instructions
        llm: Any = self.llm
        # Same dispatch as GraphRAG: message lists for V2 and LangChain models.
        if isinstance(llm, LLMInterfaceV2) or _is_langchain_chat_model(llm):
            messages = legacy_inputs_to_messages(prompt, system_instruction=system)
            response = llm.invoke(input=messages)
        else:
            response = llm.invoke(prompt, system_instruction=system)
        return str(response.content)

    def generate(self, query_text: str) -> SlotGeneration:
        prompt = self.template.format(query_text=query_text, max_slots=self.max_slots)
        content = self._invoke(prompt)
        try:
            slots = self._parse(content)
        except (ValueError, TypeError) as e:
            logger.warning("Slot answer could not be parsed (%s)", e)
            slots = []
        if not slots:
            logger.warning("No valid slots generated: using the question as slot")
            return SlotGeneration([Slot(name="answer", description=query_text)], True)
        return SlotGeneration(slots[: self.max_slots])

    @staticmethod
    def _parse(content: str) -> list[Slot]:
        text = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.M).strip()
        data: Any = json.loads(text)
        if isinstance(data, dict):
            data = data.get("slots", [])
        slots: list[Slot] = []
        for i, item in enumerate(data if isinstance(data, list) else []):
            if isinstance(item, dict) and item.get("description"):
                name = str(item.get("name") or f"slot_{i}")
                slots.append(Slot(name=name, description=str(item["description"])))
        return slots


def _is_langchain_chat_model(llm: Any) -> bool:
    try:
        from langchain_core.language_models import BaseChatModel
    except ImportError:
        return False
    return isinstance(llm, BaseChatModel)
