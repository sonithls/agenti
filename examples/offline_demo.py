"""Exercise the package with fixed fixture responses; this demo does NOT call AI."""

import json

from importance_agent import Criterion, ImportanceAgent, ProviderAdapter, Rubric


class FixtureProvider(ProviderAdapter):
    def complete(self, messages, *, schema):
        return json.dumps({
            "status": "ok",
            "explanation": "Fixture example: a supplier shutdown creates a near-term operational concern.",
            "criteria": [
                {"id": name, "rating": rating,
                 "reason": "Fixed demonstration rating, not an AI judgment.",
                 "evidence": ["shutdown starts tomorrow"]}
                for name, rating in (("relevance", 90), ("impact", 80), ("urgency", 60))
            ],
            "missing_information": ["The degree of dependence on this supplier is unspecified."],
        })

    def provenance(self):
        return {"provider": "offline-fixture", "model": "no-ai-model", "configuration_id": "demo-v1"}


def main():
    rubric = Rubric(
        instructions="Assess operational relevance, impact, and urgency over the next seven days.",
        criteria=tuple(
            Criterion(name, "Assess " + name, weight, {
                "1": "Negligible", "25": "Low", "50": "Moderate", "75": "High", "100": "Critical",
            })
            for name, weight in (("relevance", .4), ("impact", .35), ("urgency", .25))
        ),
    )
    agent = ImportanceAgent(provider=FixtureProvider(), rubric=rubric)
    result = agent.evaluate("The supplier shutdown starts tomorrow and lasts five days.")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
