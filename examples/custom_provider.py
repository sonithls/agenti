"""Bridge a different API format or SDK into ImportanceAgent.

Use CallbackProvider(request=my_request, model="my-model", configuration_id="v1")
as the agent's provider. my_request receives (messages, schema), calls your API,
and returns its generated JSON text. It must enforce network timeouts itself.
"""

from typing import Any, Callable

from importance_agent import ImportanceAgentError, ProviderAdapter, ProviderError


class CallbackProvider(ProviderAdapter):
    def __init__(self, *, request: Callable[[list[dict[str, str]], dict[str, Any]], str],
                 model: str, configuration_id: str, name: str = "custom"):
        self.request = request
        self.model = model
        self.configuration_id = configuration_id
        self.name = name

    def complete(self, messages: list[dict[str, str]], *, schema: dict[str, Any]) -> str:
        try:
            return self.request(messages, schema)
        except ImportanceAgentError:
            raise
        except Exception:
            # Do not expose arbitrary SDK exceptions containing credentials or data.
            raise ProviderError("Custom provider request failed.") from None

    def provenance(self) -> dict[str, str]:
        return {"provider": self.name, "model": self.model, "configuration_id": self.configuration_id}
