"""Custom adapters implement this small synchronous contract."""

from abc import ABC, abstractmethod
from typing import Any


class ProviderAdapter(ABC):
    """Implementations must raise package errors and bound their own I/O.

    complete() returns ONLY the generated text, not the provider's HTTP envelope.
    schema is also embedded in prompts, so native schema support is optional.
    Adapters used concurrently by callers must provide their own synchronization.
    """

    @abstractmethod
    def complete(self, messages: list[dict[str, str]], *, schema: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def provenance(self) -> dict[str, str]:
        """Return provider, model, and configuration_id; never return credentials."""
        raise NotImplementedError
