"""Local validation of input and untrusted model output."""

from __future__ import annotations

import json
import math
from typing import Any

from .errors import (
    InputTooLargeError, InsufficientInformationError, InvalidInputError, InvalidResponseError,
)
from .models import CriterionResult, Rubric


def parse_json_object(raw: str) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError
        return number

    if not isinstance(raw, str) or len(raw) > 1_000_000:
        raise InvalidResponseError("Generated output must be text of at most 1000000 characters.")
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant,
                           parse_float=finite_float)
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (ValueError, TypeError, RecursionError):
        raise InvalidResponseError("Expected one JSON object without duplicate keys or non-finite numbers.") from None


def _json_leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if value is None:
        return []
    if type(value) in (int, float, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise InvalidInputError("Input numbers must be finite.")
        return [json.dumps(value)]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise InvalidInputError("Input dictionary keys must be strings.")
        return [leaf for item in value.values() for leaf in _json_leaves(item)]
    if isinstance(value, list):
        return [leaf for item in value for leaf in _json_leaves(item)]
    raise InvalidInputError("Input must contain only JSON-compatible values.")


def normalize_input(data: str | dict[str, Any], max_chars: int) -> tuple[str, tuple[str, ...]]:
    if not isinstance(data, (str, dict)):
        raise InvalidInputError("Data must be text or a JSON-compatible dictionary.")
    try:
        leaves = _json_leaves(data)
        if not leaves:
            raise InvalidInputError("Data must contain nonempty information.")
        text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
        text.encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise InvalidInputError("Data must be valid UTF-8 text or finite, non-circular JSON data.") from None
    if len(text) > max_chars:
        raise InputTooLargeError(f"Data exceeds the configured limit of {max_chars} characters.")
    return text, tuple(leaves)


def normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise InvalidInputError("metadata must be a JSON-compatible dictionary.")
    if not metadata:
        return {}
    # Metadata can contain null/empty values; it is not evidence for scoring.
    try:
        _json_leaves(metadata)
        raw = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        raw.encode("utf-8")
        if len(raw) > 8000:
            raise InputTooLargeError("Metadata exceeds 8000 characters.")
        return json.loads(raw)
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise InvalidInputError("metadata must contain valid finite JSON data.") from None


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(s, str) and s.strip() for s in value):
        raise InvalidResponseError(f"{name} must be an array of nonempty strings.")
    return tuple(value)


def validate_evaluation(raw: str, rubric: Rubric, evidence_sources: tuple[str, ...]):
    value = parse_json_object(raw)
    if set(value) != {"status", "explanation", "criteria", "missing_information"}:
        raise InvalidResponseError("Response fields must be status, explanation, criteria, and missing_information.")
    if value["status"] not in ("ok", "insufficient_information"):
        raise InvalidResponseError("Invalid evaluation status.")
    explanation = value["explanation"]
    if not isinstance(explanation, str) or not explanation.strip():
        raise InvalidResponseError("explanation must be nonempty text.")
    missing = _strings(value["missing_information"], "missing_information")
    if not isinstance(value["criteria"], list):
        raise InvalidResponseError("criteria must be an array.")
    if value["status"] == "insufficient_information":
        if not missing or value["criteria"]:
            raise InvalidResponseError("Insufficient-information responses need gaps and an empty criteria array.")
        raise InsufficientInformationError(missing)
    if missing and rubric.missing_information_policy == "error":
        raise InsufficientInformationError(missing)
    expected = {c.id: c for c in rubric.criteria}
    results = {}
    for item in value["criteria"]:
        if not isinstance(item, dict) or set(item) != {"id", "rating", "reason", "evidence"}:
            raise InvalidResponseError("Each criterion must contain only id, rating, reason, and evidence.")
        identifier = item["id"]
        if not isinstance(identifier, str) or identifier not in expected or identifier in results:
            raise InvalidResponseError("Response contains an unknown or duplicate criterion ID.")
        if type(item["rating"]) is not int or not 1 <= item["rating"] <= 100:
            raise InvalidResponseError("Criterion ratings must be integers from 1 to 100.")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise InvalidResponseError("Each criterion needs a nonempty reason.")
        evidence = _strings(item["evidence"], "evidence")
        if not evidence and not missing:
            raise InvalidResponseError("A criterion without evidence must disclose missing information.")
        if any(not any(excerpt in source for source in evidence_sources) for excerpt in evidence):
            raise InvalidResponseError("Evidence must be exact excerpts from the supplied data values.")
        results[identifier] = CriterionResult(identifier, item["rating"], expected[identifier].weight,
                                               item["reason"], evidence)
    if set(results) != set(expected):
        raise InvalidResponseError("Response must rate every rubric criterion exactly once.")
    return explanation, tuple(results[c.id] for c in rubric.criteria), missing
