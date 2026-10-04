"""Deterministic arithmetic; never trust a model-generated overall score."""

from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence

from .errors import InvalidResponseError
from .models import CriterionResult


def calculate_importance(criteria: Sequence[CriterionResult]) -> int:
    if not criteria:
        raise InvalidResponseError("At least one criterion rating is required.")
    total = Decimal(0)
    weight_sum = Decimal(0)
    for criterion in criteria:
        if type(criterion.rating) is not int or not 1 <= criterion.rating <= 100:
            raise InvalidResponseError("Criterion ratings must be integers from 1 to 100.")
        weight = Decimal(str(criterion.weight))
        if not weight.is_finite() or weight <= 0:
            raise InvalidResponseError("Criterion weights must be finite and positive.")
        weight_sum += weight
        total += weight * criterion.rating
    return int((total / weight_sum).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
