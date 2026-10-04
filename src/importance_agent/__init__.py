"""Instruction-driven importance scoring with configurable AI providers."""

from .agent import ImportanceAgent
from .errors import (
    AuthenticationError, ConfigurationError, ImportanceAgentError, InputTooLargeError,
    InsufficientInformationError, InvalidInputError, InvalidResponseError, ProviderError,
    ProviderTimeoutError, RateLimitError, RubricError, TransportError,
)
from .models import BatchItemResult, Criterion, CriterionResult, EvaluationResult, ProviderConfig, Rubric
from .providers import ChatProvider, ProviderAdapter

__version__ = "0.1.0"
__all__ = [
    "ImportanceAgent", "ProviderConfig", "ProviderAdapter", "ChatProvider", "Criterion",
    "Rubric", "CriterionResult", "EvaluationResult", "BatchItemResult", "ImportanceAgentError",
    "AuthenticationError", "ConfigurationError", "InputTooLargeError",
    "InsufficientInformationError", "InvalidInputError", "InvalidResponseError",
    "ProviderError", "ProviderTimeoutError", "RateLimitError", "RubricError", "TransportError",
]
