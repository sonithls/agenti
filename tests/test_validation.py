import copy
import json
import unittest

from importance_agent import (
    ConfigurationError, Criterion, InputTooLargeError, InsufficientInformationError,
    InvalidInputError, InvalidResponseError, ProviderConfig, Rubric, RubricError,
)
from importance_agent.validation import normalize_input, normalize_metadata, parse_json_object, validate_evaluation
from tests.helpers import DATA, evaluation_response, make_rubric


class InputTests(unittest.TestCase):
    def test_rejects_empty_and_non_json_data(self):
        circular = {}
        circular["self"] = circular
        for value in ("", " \n", {}, {"body": ""}, {"body": None}, [], 42,
                      {1: "x"}, {"x": float("nan")}, {"x": (1, 2)}, circular, "\ud800"):
            with self.subTest(value=type(value)), self.assertRaises(InvalidInputError):
                normalize_input(value, 1000)

    def test_rejects_oversized_input(self):
        with self.assertRaises(InputTooLargeError):
            normalize_input("a" * 101, 100)

    def test_preserves_unicode_and_original_json_string_evidence(self):
        value = {"body": 'Café says "closed"\ntomorrow', "count": 0}
        text, leaves = normalize_input(value, 1000)
        self.assertEqual(json.loads(text), value)
        self.assertIn(value["body"], leaves)
        self.assertIn("0", leaves)

    def test_metadata_is_independent_and_bounded(self):
        original = {"source": {"id": 1}, "published_at": None}
        copied = normalize_metadata(original)
        original["source"]["id"] = 2
        self.assertEqual(copied["source"]["id"], 1)
        with self.assertRaises(InputTooLargeError):
            normalize_metadata({"x": "a" * 8001})


class RubricTests(unittest.TestCase):
    def test_roundtrip_and_stable_identity(self):
        rubric = make_rubric()
        restored = Rubric.from_dict(json.loads(json.dumps(rubric.to_dict())))
        self.assertEqual(rubric.rubric_id, restored.rubric_id)
        changed = rubric.to_dict()
        changed["instructions"] += " Prioritize local events."
        self.assertNotEqual(rubric.rubric_id, Rubric.from_dict(changed).rubric_id)

    def test_invalid_weights_duplicate_ids_and_anchors(self):
        base = make_rubric().to_dict()
        variants = []
        for weight in (0, -1, True, float("nan"), 0.1):
            value = copy.deepcopy(base)
            value["criteria"][0]["weight"] = weight
            variants.append(value)
        value = copy.deepcopy(base)
        value["criteria"][1]["id"] = "relevance"
        variants.append(value)
        value = copy.deepcopy(base)
        del value["criteria"][0]["anchors"]["100"]
        variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(RubricError):
                Rubric.from_dict(value)

    def test_rubric_anchors_cannot_be_mutated(self):
        anchors = {str(n): str(n) for n in (1, 25, 50, 75, 100)}
        criterion = Criterion("impact", "Assess impact", 1, anchors)
        anchors["1"] = "changed"
        self.assertEqual(criterion.anchors["1"], "1")
        with self.assertRaises(TypeError):
            criterion.anchors["1"] = "changed"


class ResponseTests(unittest.TestCase):
    def validate(self, value, rubric=None):
        return validate_evaluation(json.dumps(value), rubric or make_rubric(), (DATA,))

    def test_valid_response_and_rubric_order(self):
        value = evaluation_response()
        value["criteria"].reverse()
        _, criteria, missing = self.validate(value)
        self.assertEqual([c.id for c in criteria], ["relevance", "impact", "urgency"])
        self.assertEqual(criteria[0].weight, .4)
        self.assertEqual(missing, ())

    def test_rejects_wrong_fields_weights_ids_and_ratings(self):
        variants = []
        for key, field in (("rating", True), ("rating", 101), ("rating", 5.5),
                           ("id", "unknown"), ("weight", 1), ("reason", "")):
            value = evaluation_response()
            value["criteria"][0][key] = field
            variants.append(value)
        value = evaluation_response()
        value["importance"] = 100
        variants.append(value)
        value = evaluation_response()
        value["criteria"].pop()
        variants.append(value)
        value = evaluation_response()
        value["criteria"].append(value["criteria"][0])
        variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(InvalidResponseError):
                self.validate(value)

    def test_rejects_fabricated_evidence(self):
        value = evaluation_response()
        value["criteria"][0]["evidence"] = ["The factory exploded"]
        with self.assertRaises(InvalidResponseError):
            self.validate(value)

    def test_no_evidence_requires_disclosed_gap(self):
        value = evaluation_response()
        value["criteria"][0]["evidence"] = []
        with self.assertRaises(InvalidResponseError):
            self.validate(value)
        value["missing_information"] = ["Supplier relationship is unknown"]
        self.validate(value)
        with self.assertRaises(InsufficientInformationError):
            self.validate(value, make_rubric("error"))

    def test_insufficient_information_never_becomes_a_score(self):
        value = {"status": "insufficient_information", "criteria": [],
                 "explanation": "Cannot establish operational relevance.",
                 "missing_information": ["Which supplier the user depends on"]}
        with self.assertRaises(InsufficientInformationError) as caught:
            self.validate(value)
        self.assertEqual(caught.exception.missing_information, tuple(value["missing_information"]))

    def test_strict_json_parsing(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e400}',
                    '[]', '```json\n{}\n```', '{} trailing'):
            with self.subTest(raw=raw), self.assertRaises(InvalidResponseError):
                parse_json_object(raw)


class ConfigTests(unittest.TestCase):
    def test_invalid_configuration(self):
        for changes in (
            {"base_url": "file:///tmp/test"}, {"base_url": "https://user:secret@example.com"},
            {"base_url": "https://example.com?key=secret"}, {"base_url": "https://example.com:bad"},
            {"model": ""}, {"timeout_seconds": 0}, {"max_retries": True},
            {"max_retries": 11}, {"temperature": float("nan")}, {"structured_output": "auto"},
            {"extra_body": {"model": "override"}}, {"extra_body": {"tools": []}},
            {"api_key_env": "actual key with spaces"}, {"endpoint_path": "//other.example/api?key=x"},
        ):
            args = {"base_url": "https://example.com/v1", "model": "test"}
            args.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ConfigurationError):
                ProviderConfig(**args)
