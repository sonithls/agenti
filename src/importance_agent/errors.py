"""Public errors. Provider response bodies and credentials are never included."""


class ImportanceAgentError(Exception):
    """Base class for expected package failures."""


class ConfigurationError(ImportanceAgentError):
    pass


class InvalidInputError(ImportanceAgentError):
    pass


class InputTooLargeError(InvalidInputError):
    pass


class RubricError(ImportanceAgentError):
    pass


class InvalidResponseError(ImportanceAgentError):
    pass


class InsufficientInformationError(ImportanceAgentError):
    def __init__(self, missing_information: tuple[str, ...]):
        super().__init__("Insufficient information to evaluate under this rubric.")
        self.missing_information = missing_information


class ProviderError(ImportanceAgentError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(ProviderError):
    pass


class RateLimitError(ProviderError):
    pass


class TransportError(ProviderError):
    pass


class ProviderTimeoutError(TransportError):
    pass
