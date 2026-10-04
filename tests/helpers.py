import json

from importance_agent import Criterion, ProviderAdapter, Rubric

INSTRUCTIONS = "Evaluate operational relevance, impact, and urgency over the next seven days."
DATA = "The supplier shutdown starts tomorrow and lasts five days."


def make_rubric(policy="evaluate_available"):
    return Rubric(
        instructions=INSTRUCTIONS,
        criteria=tuple(
            Criterion(name, "Assess " + name, weight,
                      {str(n): f"{name} importance at {n}" for n in (1, 25, 50, 75, 100)})
            for name, weight in (("relevance", 0.4), ("impact", 0.35), ("urgency", 0.25))
        ),
        missing_information_policy=policy,
    )


def rubric_response():
    value = make_rubric().to_dict()
    return json.dumps({key: value[key] for key in ("criteria", "missing_information_policy")})


def evaluation_response():
    return {
        "status": "ok", "explanation": "A near-term supplier interruption affects operations.",
        "criteria": [
            {"id": name, "rating": rating, "reason": "Supplied information supports this assessment.",
             "evidence": ["shutdown starts tomorrow"]}
            for name, rating in (("relevance", 90), ("impact", 80), ("urgency", 60))
        ],
        "missing_information": [],
    }


class FakeProvider(ProviderAdapter):
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, *, schema):
        self.calls.append((messages, schema))
        if not self.responses:
            raise AssertionError("Unexpected additional provider call")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, str) else json.dumps(value)

    def provenance(self):
        return {"provider": "fake", "model": "fixture", "configuration_id": "fixture-v1"}
