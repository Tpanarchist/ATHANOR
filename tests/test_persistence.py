import hashlib
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from athanor import Athanor, Endpoint, FieldSpec, PersistenceError, Procedure, Role
from athanor.demo import run_demo
from athanor.programs import argument, binary, field, literal


def rehash(envelope):
    payload = json.dumps(
        envelope["payload"],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    envelope["sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return json.dumps(envelope)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.a = Athanor(namespace="persistence")
        self.category = self.a.teach_category("Number", {"value": FieldSpec("number", minimum=0)})
        self.ref = self.a.create(self.category, {"value": Fraction(1, 3)})
        self.operation = self.a.teach_operation(
            "increment",
            Procedure(
                (Endpoint(Role.DATA, self.category),),
                self.category,
                {"value": binary("add", field(argument(0), "value"), literal(1))},
                update=0,
            ),
        )
        self.changed = self.a.execute(self.operation, (self.ref,))

    def corrupted(self, mutation):
        envelope = json.loads(self.a.dumps())
        mutation(envelope)
        with self.assertRaises(PersistenceError):
            Athanor.loads(rehash(envelope))

    def test_canonical_roundtrip_and_restored_execution(self):
        snapshot = self.a.dumps()
        restored = Athanor.loads(snapshot)
        self.assertEqual(restored.dumps(), snapshot)
        self.assertEqual(restored.structures, self.a.structures)
        self.assertEqual(restored.receipts, self.a.receipts)
        self.assertEqual(restored.get(self.ref).fields["value"], Fraction(1, 3))
        self.assertEqual(restored.get(self.changed).fields["value"], Fraction(4, 3))
        next_ref = restored.execute(self.operation, (self.changed,))
        self.assertEqual(restored.get(next_ref).fields["value"], Fraction(7, 3))

    def test_same_explicit_namespace_and_actions_reproduce_identical_snapshots(self):
        def build():
            a = Athanor(namespace="reproducible")
            category = a.teach_category("Number", {"value": FieldSpec("number")})
            a.create(category, {"value": Fraction(2, 7)})
            return a.dumps()

        self.assertEqual(build(), build())

    def test_every_value_kind_roundtrips_without_tag_collisions(self):
        ref = self.a.create(
            fields={
                "null": None,
                "bool": True,
                "text": "α",
                "rational": Fraction(-1, 7),
                "ref": self.ref,
                "sequence": [1, "two"],
                "looks_like_tag": {"op": "ref", "id": "s99999999"},
            }
        )
        restored = Athanor.loads(self.a.dumps())
        self.assertEqual(restored.get(ref), self.a.get(ref))

    def test_empty_store_roundtrips(self):
        a = Athanor(namespace="empty")
        self.assertEqual(Athanor.loads(a.dumps()).dumps(), a.dumps())

    def test_file_save_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.json"
            self.a.save(path)
            self.assertEqual(Athanor.restore(path).dumps(), self.a.dumps())

    def test_failed_replace_preserves_existing_snapshot_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "knowledge.json"
            path.write_text("old", encoding="utf-8")
            with patch("athanor.persistence.os.replace", side_effect=OSError("simulated failure")):
                with self.assertRaises(PersistenceError):
                    self.a.save(path)
            self.assertEqual(path.read_text(encoding="utf-8"), "old")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_missing_file_and_missing_parent_are_explicit_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PersistenceError):
                Athanor.restore(Path(directory) / "missing.json")
            with self.assertRaises(PersistenceError):
                self.a.save(Path(directory) / "missing" / "state.json")

    def test_checksum_mismatch(self):
        envelope = json.loads(self.a.dumps())
        envelope["payload"]["namespace"] = "changed"
        with self.assertRaisesRegex(PersistenceError, "digest mismatch"):
            Athanor.loads(json.dumps(envelope))

    def test_duplicate_json_keys_invalid_json_and_nonfinite_constants(self):
        for raw in ('{"payload":{},"payload":{}}', '{"payload":NaN}', "not json"):
            with self.assertRaises(PersistenceError):
                Athanor.loads(raw)

    def test_unknown_versions_and_extra_top_level_fields(self):
        self.corrupted(lambda env: env["payload"].update(version=2))
        self.corrupted(lambda env: env["payload"].update(version=True))
        self.corrupted(lambda env: env.update(extra="not allowed"))

    def test_rehashed_foreign_reference_is_rejected(self):
        def mutate(env):
            env["payload"]["entries"][1]["structure"]["category"][1] = "foreign"

        self.corrupted(mutate)

    def test_rehashed_missing_reference_and_forward_reference_are_rejected(self):
        def mutate(env):
            env["payload"]["entries"][1]["structure"]["category"][2] = "s99999999"

        self.corrupted(mutate)

        def forward(env):
            env["payload"]["entries"][0]["structure"]["fields"][1][1][1] = [
                "sequence",
                [["ref", "persistence", "s00000001", 1]],
            ]

        self.corrupted(forward)

    def test_rehashed_invalid_required_field_type_is_rejected(self):
        def mutate(env):
            env["payload"]["entries"][1]["structure"]["fields"][1][0][1] = ["bool", True]

        self.corrupted(mutate)

    def test_rehashed_noncanonical_rational_and_record_keys_are_rejected(self):
        def mutate(env):
            env["payload"]["entries"][1]["structure"]["fields"][1][0][1] = ["rational", "2", "6"]

        self.corrupted(mutate)

        def duplicate(env):
            fields = env["payload"]["entries"][1]["structure"]["fields"][1]
            fields.append(fields[0])

        self.corrupted(duplicate)

    def test_rehashed_receipt_sequence_result_and_action_are_rejected(self):
        self.corrupted(lambda env: env["payload"]["entries"][0]["receipt"].update(sequence=2))
        self.corrupted(lambda env: env["payload"]["entries"][0]["receipt"].update(sequence=True))
        self.corrupted(lambda env: env["payload"]["entries"][0]["receipt"].update(action="execute"))

        def result(env):
            env["payload"]["entries"][0]["receipt"]["result"][2] = "s00000002"

        self.corrupted(result)

    def test_rehashed_revision_and_causal_future_are_rejected(self):
        def revision(env):
            entry = env["payload"]["entries"][-1]
            entry["structure"]["ref"][3] = 3
            entry["receipt"]["result"][3] = 3

        self.corrupted(revision)
        self.corrupted(lambda env: env["payload"]["entries"][-1]["receipt"].update(causes=[100]))

    def test_rehashed_operation_grammar_and_interface_are_rejected(self):
        def grammar(env):
            # An operation's procedure fields contain expression records, not source code.
            operation = env["payload"]["entries"][2]["structure"]["fields"][1]
            procedure = dict(operation)["procedure"][1]
            fields = dict(procedure)["fields"][1]
            expression = fields[0][1][1]
            for pair in expression:
                if pair[0] == "op":
                    pair[1] = ["string", "execute_python"]

        self.corrupted(grammar)
        self.corrupted(
            lambda env: env["payload"]["entries"][-1]["receipt"].update(inputs=["sequence", []])
        )

    def test_teaching_after_restore_survives_another_restore(self):
        restored = Athanor.loads(self.a.dumps())
        operation = restored.teach_operation(
            "half",
            Procedure(
                (Endpoint(Role.DATA, self.category),),
                self.category,
                {"value": binary("div", field(argument(0), "value"), literal(2))},
            ),
        )
        again = Athanor.loads(restored.dumps())
        result = again.execute(operation, (self.changed,))
        self.assertEqual(again.get(result).fields["value"], Fraction(2, 3))

    def test_end_to_end_demo(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_demo(Path(directory) / "demo.json")
            self.assertEqual(report["recursive_total"], "11/2")
            self.assertEqual(report["restored_total"], "11/2")
            self.assertEqual(report["new_taught_procedure_result"], "11")
            self.assertEqual(report["second_restore_result"], "11")
            self.assertEqual(report["inherited_claim_count"], 1)
            self.assertEqual(report["unasserted_applicability"], "unknown")
            self.assertEqual(report["disputed_applicability"], "conflict")


if __name__ == "__main__":
    unittest.main()
