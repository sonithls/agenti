"""Validated configuration, immutable rubrics, and serializable results."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit

from .errors import ConfigurationError, ImportanceAgentError, RubricError


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    model: str
    api_key_env: str | None = "IMPORTANCE_API_KEY"
    adapter: str = "chat"
    endpoint_path: str = "/chat/completions"
    timeout_seconds: float = 30
    max_retries: int = 2
    max_retry_delay_seconds: float = 30
    max_response_bytes: int = 1_000_000
    temperature: float | None = 0
    structured_output: str = "prompt"
    extra_body: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        try:
            url = urlsplit(self.base_url)
            valid_url = (
                url.scheme in ("http", "https") and url.hostname and not url.username
                and not url.password and not url.query and not url.fragment
                and not any(c.isspace() for c in self.base_url)
            )
            _ = url.port
        except (TypeError, ValueError, AttributeError):
            valid_url = False
        if not valid_url:
            raise ConfigurationError("base_url must be an HTTP(S) URL without credentials, query, or fragment.")
        if not _text(self.model) or not _text(self.adapter):
            raise ConfigurationError("model and adapter must be nonempty strings.")
        if self.api_key_env is not None and (
            not isinstance(self.api_key_env, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env)
        ):
            raise ConfigurationError("api_key_env must name an environment variable, or be None.")
        if not isinstance(self.endpoint_path, str) or not re.fullmatch(
            r"/[A-Za-z0-9_/-]+", self.endpoint_path
        ):
            raise ConfigurationError("endpoint_path must be an absolute API path.")
        for name in ("timeout_seconds", "max_retry_delay_seconds"):
            if not _number(getattr(self, name)) or getattr(self, name) <= 0:
                raise ConfigurationError(f"{name} must be a finite positive number.")
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 10:
            raise ConfigurationError("max_retries must be an integer between 0 and 10.")
        if type(self.max_response_bytes) is not int or self.max_response_bytes < 1:
            raise ConfigurationError("max_response_bytes must be a positive integer.")
        if self.temperature is not None and (
            not _number(self.temperature) or not 0 <= self.temperature <= 2
        ):
            raise ConfigurationError("temperature must be between 0 and 2, or None to omit it.")
        if self.structured_output not in ("prompt", "json_object", "json_schema"):
            raise ConfigurationError("structured_output must be prompt, json_object, or json_schema.")
        reserved = {"model", "messages", "stream", "n", "tools", "tool_choice", "functions",
                    "function_call", "response_format", "temperature"}
        if not isinstance(self.extra_body, Mapping) or reserved.intersection(self.extra_body):
            raise ConfigurationError("extra_body must be a mapping and cannot override reserved request fields.")
        try:
            if any(not isinstance(k, str) for k in self.extra_body):
                raise ValueError
            copied = json.loads(json.dumps(dict(self.extra_body), allow_nan=False))
        except (TypeError, ValueError, OverflowError, RecursionError):
            raise ConfigurationError("extra_body must contain finite JSON-compatible values.") from None
        object.__setattr__(self, "extra_body", MappingProxyType(copied))

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + self.endpoint_path


@dataclass(frozen=True)
class Criterion:
    id: str
    description: str
    weight: float
    anchors: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.id):
            raise RubricError("Criterion IDs must be lowercase identifiers of at most 64 characters.")
        if not _text(self.description):
            raise RubricError("Every criterion needs a description.")
        if not _number(self.weight) or not 0 < self.weight <= 1:
            raise RubricError("Criterion weights must be finite numbers greater than 0 and at most 1.")
        if not isinstance(self.anchors, Mapping) or set(self.anchors) != {"1", "25", "50", "75", "100"}:
            raise RubricError("Every criterion needs anchors with string keys 1, 25, 50, 75, and 100.")
        if not all(_text(value) for value in self.anchors.values()):
            raise RubricError("Each rating anchor must have a nonempty description.")
        object.__setattr__(self, "anchors", MappingProxyType(dict(self.anchors)))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "description": self.description, "weight": self.weight,
                "anchors": dict(self.anchors)}


@dataclass(frozen=True)
class Rubric:
    instructions: str
    criteria: tuple[Criterion, ...]
    missing_information_policy: str = "evaluate_available"
    version: int = 1

    def __post_init__(self) -> None:
        if not _text(self.instructions) or len(self.instructions) > 10_000:
            raise RubricError("instructions must contain 1..10000 characters of nonempty text.")
        if not isinstance(self.criteria, (list, tuple)) or not 1 <= len(self.criteria) <= 20:
            raise RubricError("A rubric needs between 1 and 20 criteria.")
        if not all(isinstance(c, Criterion) for c in self.criteria):
            raise RubricError("criteria must contain Criterion objects.")
        if len({c.id for c in self.criteria}) != len(self.criteria):
            raise RubricError("Criterion IDs must be unique.")
        if not math.isclose(sum(c.weight for c in self.criteria), 1, rel_tol=0, abs_tol=1e-6):
            raise RubricError("Criterion weights must sum to 1 within an absolute tolerance of 0.000001.")
        if self.missing_information_policy not in ("evaluate_available", "error"):
            raise RubricError("missing_information_policy must be evaluate_available or error.")
        if type(self.version) is not int or self.version != 1:
            raise RubricError("Only rubric format version 1 is supported.")
        object.__setattr__(self, "criteria", tuple(self.criteria))
        if len(json.dumps(self.to_dict(), ensure_ascii=False)) > 50_000:
            raise RubricError("Serialized rubric must not exceed 50000 characters.")

    @property
    def rubric_id(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return "rubric_" + hashlib.sha256(canonical.encode()).hexdigest()[:20]

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "instructions": self.instructions,
                "criteria": [c.to_dict() for c in self.criteria],
                "missing_information_policy": self.missing_information_policy}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Rubric:
        try:
            if set(value) != {"version", "instructions", "criteria", "missing_information_policy"}:
                raise ValueError
            return cls(instructions=value["instructions"],
                       criteria=tuple(Criterion(**c) for c in value["criteria"]),
                       missing_information_policy=value["missing_information_policy"],
                       version=value["version"])
        except (TypeError, KeyError, ValueError):
            raise RubricError("Invalid serialized rubric structure.") from None


@dataclass(frozen=True)
class CriterionResult:
    id: str
    rating: int
    weight: float
    reason: str
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "rating": self.rating, "weight": self.weight,
                "reason": self.reason, "evidence": list(self.evidence)}


@dataclass(frozen=True)
class EvaluationResult:
    importance: int
    explanation: str
    criteria: tuple[CriterionResult, ...]
    missing_information: tuple[str, ...]
    rubric_id: str
    provider: str
    model: str
    evaluated_at: str
    configuration_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"importance": self.importance, "explanation": self.explanation,
                "criteria": [c.to_dict() for c in self.criteria],
                "missing_information": list(self.missing_information),
                "rubric_id": self.rubric_id, "provider": self.provider, "model": self.model,
                "evaluated_at": self.evaluated_at, "configuration_id": self.configuration_id,
                "metadata": dict(self.metadata)}


@dataclass(frozen=True)
class BatchItemResult:
    index: int
    result: EvaluationResult | None = None
    error: ImportanceAgentError | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        error = None
        if self.error is not None:
            error = {"type": type(self.error).__name__, "message": str(self.error)}
            if hasattr(self.error, "missing_information"):
                error["missing_information"] = list(self.error.missing_information)
        return {"index": self.index, "ok": self.ok,
                "result": self.result.to_dict() if self.result else None, "error": error}
