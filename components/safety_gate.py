"""Deterministic safety gate for the Eco-Travel Advisor NLU pipeline.

Why it exists
-------------
DIET is a statistical classifier: it can miss a visa or health question that
is phrased in a way it has not seen, and then the bot answers with travel tips
instead of saying "check an official source". The lecturer's chatbot pipeline
puts a deterministic check in front of the model for exactly this reason:
"Safety controls should not depend only on a generator following
instructions" (NLP-LLM §17), and a small regex gate routes selected high-risk
categories to a fixed response. Like the lecturer's gate, this one is
"intentionally conservative and incomplete": it only adds a guaranteed path
for the patterns it lists; everything else is still classified by DIET.

It also performs input validation, the first step of the lecturer's pipeline
table ("Rejects empty or excessively long inputs", NLP-LLM §1): an empty or
over-long message is mapped to the `invalid_input` intent, which has a fixed
reply.

How it works
------------
It runs after FallbackClassifier. For each message it
  1. maps empty / whitespace-only / over-long text to `invalid_input`;
  2. otherwise checks the lowercased text against the regex patterns in
     `patterns_file`, category by category, in file order. The first match
     sets that category's intent with confidence 1.0 and puts it at the top
     of the intent ranking.
The matched category is stored under `message.data["safety_gate"]` so the
decision is visible in the tracker (an auditable trace, NLP-LLM §1).

The patterns file is read when the model is loaded, not at training time, so
it must be present next to `config.yml` wherever the bot runs.

Patterns file format (`data/safety_patterns.yml`)::

    categories:
      - name: regulated_advice          # label shown in the trace
        intent: ask_regulated_advice    # intent the gate sets
        patterns:                       # Python regexes, matched lowercase
          - '\\bvisas?\\b'
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Tuple

import yaml

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.shared.nlu.constants import (
    INTENT,
    INTENT_NAME_KEY,
    INTENT_RANKING_KEY,
    PREDICTED_CONFIDENCE_KEY,
    TEXT,
)
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)

INVALID_INPUT_INTENT = "invalid_input"
TRACE_KEY = "safety_gate"


def load_categories(path: Path) -> List[Tuple[Text, Text, List[re.Pattern]]]:
    """Read and compile the pattern categories. A missing file means no
    patterns (input validation still works); a malformed one raises."""
    if not path.exists():
        logger.warning("SafetyGate: %s not found, no keyword patterns loaded.", path)
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    categories = []
    for cat in data.get("categories", []):
        compiled = [re.compile(p) for p in cat.get("patterns", [])]
        categories.append((cat["name"], cat["intent"], compiled))
    return categories


def classify(
    text: Optional[Text],
    categories: List[Tuple[Text, Text, List[re.Pattern]]],
    max_chars: int,
) -> Optional[Tuple[Text, Text]]:
    """Return (category, intent) the gate imposes on `text`, or None."""
    if text is None or not text.strip():
        return "empty_input", INVALID_INPUT_INTENT
    if len(text) > max_chars:
        return "too_long", INVALID_INPUT_INTENT
    lowered = text.lower()
    for name, intent, patterns in categories:
        if any(p.search(lowered) for p in patterns):
            return name, intent
    return None


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=False
)
class SafetyGate(GraphComponent, IntentClassifier):
    """Overrides the predicted intent for empty, over-long or high-risk text."""

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {"patterns_file": "data/safety_patterns.yml", "max_chars": 500}

    def __init__(self, config: Dict[Text, Any]) -> None:
        self.max_chars = int(config["max_chars"])
        self.categories = load_categories(Path(config["patterns_file"]))

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "SafetyGate":
        return cls(config)

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            decision = classify(message.get(TEXT), self.categories, self.max_chars)
            if decision is None:
                continue
            category, intent = decision
            imposed = {INTENT_NAME_KEY: intent, PREDICTED_CONFIDENCE_KEY: 1.0}
            message.set(INTENT, imposed, add_to_output=True)
            ranking = [
                r for r in message.get(INTENT_RANKING_KEY, []) or []
                if r.get(INTENT_NAME_KEY) != intent
            ]
            message.set(INTENT_RANKING_KEY, [imposed] + ranking, add_to_output=True)
            message.set(TRACE_KEY, category, add_to_output=True)
        return messages
