# importance-agent

A Python package that scores news or other information **1–100** according to
your evaluation instructions. It returns the score, an explanation, individual
criterion ratings, supporting excerpts, and missing information.

Python **3.10+**. No third-party runtime dependencies. Bring your own AI API
endpoint and model; API access and free-tier quotas are controlled by your provider.

## Install

From this project directory:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install .
```

Use another Python 3.10+ interpreter if preferred. The default `python3` in the
development workspace is 3.7 and cannot run this package. The package is local;
it has not been published to PyPI.

## Configure your API and evaluate

Set these environment variables in your shell or application environment:

```bash
export IMPORTANCE_BASE_URL="https://your-provider.example/v1"
export IMPORTANCE_MODEL="your-model-name"
export IMPORTANCE_API_KEY="your-api-key"
```

The URL and model above are placeholders. Supply your provider's actual values.

```python
import os
from importance_agent import ImportanceAgent, ProviderConfig

agent = ImportanceAgent(
    provider=ProviderConfig(
        base_url=os.environ["IMPORTANCE_BASE_URL"],
        model=os.environ["IMPORTANCE_MODEL"],
        api_key_env="IMPORTANCE_API_KEY",
    ),
    instructions=(
        "Evaluate importance to our manufacturing operations over the next 7 days. "
        "Use relevance at 40%, operational impact at 35%, and urgency at 25%. "
        "Explain unknowns and assess the available evidence without inventing facts."
    ),
)

result = agent.evaluate({
    "title": "Supplier announces a factory shutdown",
    "body": "The shutdown starts tomorrow and lasts five days.",
})

print(result.importance)           # Integer 1..100 on successful evaluation
print(result.explanation)
print(result.to_dict())            # JSON-serializable result
```

The first evaluation normally makes one rubric-generation call and one
evaluation call. Later evaluations on the same agent reuse that rubric.
Each generation allows at most one additional repair call for invalid output.
Transport retries are separately bounded by `max_retries` (default 2).

Input is nonempty text or a JSON-compatible dictionary. Dictionaries may contain
nested objects and lists, but top-level lists belong in `evaluate_many`.

Run the file-based example with your API configured:

```bash
.venv/bin/python examples/evaluate_news.py examples/news.json \
  --instructions "Assess operational relevance, impact, and urgency over the next 7 days."
```

The example accepts UTF-8 text files as well as JSON objects.

## API compatibility

The built-in `chat` adapter sends `POST {base_url}/chat/completions` with a model,
messages, and `stream: false`. It reads generated text from
`choices[0].message.content` and uses a bearer API key from the named environment
variable. The base URL should stop at the API prefix, not include
`/chat/completions`.

For example, Groq documents the compatible base URL
`https://api.groq.com/openai/v1`; choose a model available to your account.
See [provider compatibility documentation](https://console.groq.com/docs/openai).
This example does not guarantee free access or the availability of any model.

```python
config = ProviderConfig(
    base_url="https://your-provider.example/v1",
    model="your-model-name",
    api_key_env="MY_PROVIDER_KEY",     # None for an endpoint requiring no key
    endpoint_path="/chat/completions",
    timeout_seconds=30,
    max_retries=2,
    max_retry_delay_seconds=30,
    max_response_bytes=1_000_000,
    temperature=0,                    # None omits the field for incompatible models
    structured_output="prompt",       # Or "json_object" / "json_schema"
    extra_body={"max_tokens": 4096},   # Only parameters your provider supports
)
```

- `prompt` embeds the JSON schema in instructions and validates the answer locally.
- `json_object` also requests JSON mode from the API.
- `json_schema` also requests native strict schema output. Choose it only when
  the provider/model supports the supplied schema constraints. See the provider's
  [structured-output documentation](https://console.groq.com/docs/structured-outputs).

There is no silent fallback to another mode or provider. An unsupported request
returns a typed error so you can correct the configuration. A free API with a
different request format, authentication mechanism, or response layout needs a
custom adapter; changing the URL alone is insufficient.

Implement `ProviderAdapter.complete(messages, *, schema)` to return generated
JSON text and `provenance()` to return provider/model/configuration identifiers.
[examples/custom_provider.py](examples/custom_provider.py) provides a callback
bridge for another API or SDK. Custom adapters must enforce their own timeouts,
response limits, retries, and credential handling, and raise package errors.
If you add generation settings, change the adapter's configuration identifier.

## Review, save, or supply the scoring rubric

```python
import json
from pathlib import Path
from importance_agent import Rubric

rubric = agent.prepare()
print(json.dumps(rubric.to_dict(), indent=2))
Path("rubric.json").write_text(json.dumps(rubric.to_dict(), indent=2), encoding="utf-8")

restored = Rubric.from_dict(json.loads(Path("rubric.json").read_text(encoding="utf-8")))
another_agent = ImportanceAgent(provider=config, rubric=restored)
```

Supply your own rubric to skip AI rubric generation and define the exact weights:

```python
from importance_agent import Criterion, Rubric

rubric = Rubric(
    instructions="Evaluate importance to our operations over the next seven days.",
    criteria=(
        Criterion(
            id="operational_impact",
            description="Assess disruption to our production; disclose unknown dependencies.",
            weight=1.0,
            anchors={
                "1": "No identifiable operational effect",
                "25": "Minor inconvenience with an available workaround",
                "50": "Noticeable disruption affecting one workflow",
                "75": "Major disruption requiring management action",
                "100": "Critical interruption threatening continued operations",
            },
        ),
    ),
    missing_information_policy="evaluate_available",
)
agent = ImportanceAgent(provider=config, rubric=rubric)
```

Weights must be positive and sum to 1 within `0.000001`. Every criterion needs
anchors with string keys `"1"`, `"25"`, `"50"`, `"75"`, and `"100"`. The final score is
the weighted average, rounded half up in Python. For example, ratings 90, 80, 60
with weights 0.40, 0.35, 0.25 produce **79**.

Rubrics are immutable and have content-derived IDs. Use a new agent for changed
instructions; supplying instructions that disagree with a saved rubric is an error.
Compare scores under the same rubric and model settings. Model judgments can vary
even with low randomness. No confidence percentage is invented.

With `evaluate_available`, the model may assess available evidence and report
gaps; it can still declare an item impossible to assess. With `error`, any
reported relevant gap prevents a score. Detecting gaps remains a model judgment.

## Batches and failures

```python
items = ["First news item", {"body": "Second item"}, ""]
for item in agent.evaluate_many(items):
    if item.ok:
        print(item.index, item.result.importance)
    else:
        print(item.index, type(item.error).__name__, str(item.error))
```

Batches run sequentially, preserve input order, and isolate per-item errors.
The agent prepares one shared rubric before its first valid item; failure to
prepare that rubric raises once for the whole batch. Empty or entirely invalid
batches do not call the API. For large streams, submit bounded batches.

```python
from importance_agent import ImportanceAgentError, InsufficientInformationError

try:
    result = agent.evaluate("Some information", metadata={"item_id": "news-42"})
except InsufficientInformationError as error:
    print(error.missing_information)
except ImportanceAgentError as error:
    print(type(error).__name__, str(error))
```

Other public errors include `ConfigurationError`, `RubricError`,
`InvalidInputError`, `InputTooLargeError`, `InvalidResponseError`,
`AuthenticationError`, `RateLimitError`, `ProviderTimeoutError`, `TransportError`,
and `ProviderError`. Failures never produce a substitute importance score.

## Limits and data handling

- Default input limit: 60,000 characters, configurable with `max_input_chars`.
  Metadata is limited to 8,000 characters. Oversized input is rejected, not truncated.
  Character limits are not token limits; a model can reject smaller inputs too.
- HTTP responses are capped at 1 MB by default. Generated JSON text is also capped
  at 1,000,000 characters. Native schema support does not replace local validation.
- Timeouts apply to network operations, with an elapsed-time check between response
  chunks. A blocking read can last up to the socket timeout beyond that check;
  this is not a strict whole-evaluation deadline. Retries and repair add latency.
- Rate limits and transient HTTP/network failures receive bounded retries. A
  `Retry-After` longer than the configured wait limit returns an error immediately,
  rather than retrying before the requested time. Authentication errors are not retried.
- The package sends supplied content, instructions, and metadata to your configured
  endpoint. It does not persist input or log API keys, prompts, or response bodies.
  Result metadata includes any metadata you provide. Keep secrets out of `extra_body`;
  it is intended only for generation settings. There is no implicit `.env` loader.
- Evidence excerpts must occur in the original data values. This checks quotation
  fidelity, not whether the claims or the model's reasoning are true.
- Submitted data and metadata are kept separate from scoring instructions. This
  reduces prompt-injection risk but does not guarantee immunity. The model has
  no package-provided tools or execution access.
- Version 1 does not browse, verify sources, parse PDFs, or summarize long documents.

## Offline demo and tests

No API key or network access is needed for these commands:

```bash
.venv/bin/python examples/offline_demo.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t . -v
```

The offline demo uses fixed fixture responses to demonstrate a result of 79; it
does **not** simulate meaningful AI judgment. Automated tests cover score arithmetic,
validation, rubric reuse, repair limits, batch errors, and mocked HTTP behavior.
Scoring quality should be assessed separately with examples you have rated yourself.

The original design is in [importance_agent_architecture.txt](importance_agent_architecture.txt).
