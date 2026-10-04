"""Public orchestration API."""

from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Iterable, TypeVar

from .errors import ConfigurationError, ImportanceAgentError, InvalidInputError, InvalidResponseError, RubricError
from .models import BatchItemResult, EvaluationResult, ProviderConfig, Rubric
from .prompts import EVALUATION_SCHEMA, RUBRIC_SCHEMA, evaluation_messages, rubric_messages
from .providers import ChatProvider, ProviderAdapter
from .rubric import parse_rubric
from .scoring import calculate_importance
from .validation import normalize_input, normalize_metadata, validate_evaluation

_T = TypeVar("_T")


class ImportanceAgent:
    def __init__(self, *, provider: ProviderConfig | ProviderAdapter,
                 instructions: str | None = None, rubric: Rubric | None = None,
                 max_input_chars: int = 60_000):
        if rubric is not None and not isinstance(rubric, Rubric):
            raise ConfigurationError("rubric must be a Rubric object.")
        if instructions is None and rubric is not None:
            instructions = rubric.instructions
        if not isinstance(instructions, str) or not instructions.strip() or len(instructions) > 10_000:
            raise ConfigurationError("Provide nonempty instructions of at most 10000 characters.")
        if rubric is not None and instructions != rubric.instructions:
            raise ConfigurationError("Instructions must match the supplied rubric; create a new rubric for new instructions.")
        if type(max_input_chars) is not int or max_input_chars <= 0:
            raise ConfigurationError("max_input_chars must be a positive integer.")
        if isinstance(provider, ProviderConfig):
            if provider.adapter != "chat":
                raise ConfigurationError("Only adapter='chat' is built in; pass a custom ProviderAdapter instance for other APIs.")
            provider = ChatProvider(provider)
        if not isinstance(provider, ProviderAdapter):
            raise ConfigurationError("provider must be ProviderConfig or a ProviderAdapter instance.")
        self._provider = provider
        self._instructions = instructions
        self._rubric = rubric
        self._max_input_chars = max_input_chars
        self._lock = RLock()

    @property
    def instructions(self) -> str:
        return self._instructions

    def prepare(self) -> Rubric:
        """Build the rubric once, or return the explicitly supplied rubric."""
        with self._lock:
            if self._rubric is None:
                self._rubric = self._validated_call(
                    rubric_messages(self.instructions), RUBRIC_SCHEMA,
                    lambda raw: parse_rubric(raw, self.instructions),
                )
            return self._rubric

    def _validated_call(self, messages: list[dict[str, str]], schema: dict[str, Any],
                        validate: Callable[[str], _T]) -> _T:
        # At most one repair call; transport retries are separately bounded.
        for attempt in range(2):
            try:
                raw = self._provider.complete(messages, schema=schema)
                return validate(raw)
            except (InvalidResponseError, RubricError):
                if attempt:
                    raise InvalidResponseError("Provider failed output validation after one repair attempt.") from None
                messages = messages + [{"role": "user", "content": (
                    "The previous response failed local validation. Generate a fresh JSON object "
                    "following the original schema and instructions exactly. Check field names, "
                    "all criterion IDs, numeric ranges, weight totals if building a rubric, "
                    "and verbatim evidence if evaluating. Return no markdown or extra fields."
                )}]
        raise AssertionError("Unreachable validation state")

    def evaluate(self, data: str | dict[str, Any], *,
                 metadata: dict[str, Any] | None = None) -> EvaluationResult:
        """Evaluate one item, raising a typed error if no valid score is available."""
        text, evidence_sources = normalize_input(data, self._max_input_chars)
        copied_metadata = normalize_metadata(metadata)
        # Serialize calls through an agent, including custom adapters.
        with self._lock:
            rubric = self.prepare()
            evaluated_at = datetime.now(timezone.utc).isoformat()
            provenance = self._provider.provenance()
            if not isinstance(provenance, dict) or any(
                not isinstance(provenance.get(key), str) or not provenance[key].strip()
                for key in ("provider", "model", "configuration_id")
            ):
                raise ConfigurationError("Adapter provenance needs provider, model, and configuration_id strings.")
            explanation, criteria, missing = self._validated_call(
                evaluation_messages(rubric, text, copied_metadata, evaluated_at),
                EVALUATION_SCHEMA,
                lambda raw: validate_evaluation(raw, rubric, evidence_sources),
            )
            return EvaluationResult(
                importance=calculate_importance(criteria), explanation=explanation,
                criteria=criteria, missing_information=missing, rubric_id=rubric.rubric_id,
                provider=provenance["provider"], model=provenance["model"],
                configuration_id=provenance["configuration_id"], evaluated_at=evaluated_at,
                metadata=copied_metadata,
            )

    def evaluate_many(self, items: Iterable[str | dict[str, Any]]) -> list[BatchItemResult]:
        """Sequential batch: stable order, shared rubric, isolated per-item errors.

        A shared rubric-preparation failure raises once for the whole batch.
        No API calls are made for an empty batch.
        """
        if isinstance(items, (str, bytes, dict)):
            raise InvalidInputError("items must be an iterable of individual text/dictionary items.")
        try:
            iterator = iter(items)
        except TypeError:
            raise InvalidInputError("items must be iterable.") from None
        results = []
        prepared = False
        for index, item in enumerate(iterator):
            try:
                normalize_input(item, self._max_input_chars)
            except InvalidInputError as error:
                results.append(BatchItemResult(index=index, error=error))
                continue
            if not prepared:
                self.prepare()
                prepared = True
            try:
                results.append(BatchItemResult(index=index, result=self.evaluate(item)))
            except ImportanceAgentError as error:
                results.append(BatchItemResult(index=index, error=error))
        return results
