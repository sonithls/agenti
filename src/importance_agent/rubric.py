"""Create validated rubrics from structured model output."""

from .errors import RubricError
from .models import Criterion, Rubric
from .validation import parse_json_object


def parse_rubric(raw: str, instructions: str) -> Rubric:
    value = parse_json_object(raw)
    if set(value) != {"criteria", "missing_information_policy"} or not isinstance(value["criteria"], list):
        raise RubricError("Generated rubric must contain criteria and missing_information_policy.")
    try:
        return Rubric(instructions=instructions,
                      criteria=tuple(Criterion(**item) for item in value["criteria"]),
                      missing_information_policy=value["missing_information_policy"])
    except TypeError:
        raise RubricError("Generated rubric has an invalid criterion structure.") from None
